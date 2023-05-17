use crate::{PolyIOP, PolyIOPErrors, PolynomialCommitmentScheme, SumCheck, ZeroCheck};
use arithmetic::VirtualPolynomial;
use ark_ec::pairing::Pairing;
use ark_ff::{batch_inversion, PrimeField};
use ark_poly::DenseMultilinearExtension;
use ark_std::{One, Zero};
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use std::{collections::HashMap, iter, ops::Deref, sync::Arc};
use transcript::IOPTranscript;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct LookupCheckSubClaim<F: PrimeField, ZC: ZeroCheck<F>> {
    // the SubClaim from the ZeroCheck
    pub zero_check_sub_claim: ZC::ZeroCheckSubClaim,
    // Challenge for vector lookup
    pub theta: F,
    // Challenge for logarithmic derivative lookup
    pub tau: F,
    // Challenge for batch zero check
    pub alpha: F,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct LookupCheckProof<
    E: Pairing,
    PCS: PolynomialCommitmentScheme<E>,
    ZC: ZeroCheck<E::ScalarField>,
> {
    pub zero_check_proof: ZC::ZeroCheckProof,
    pub m_comm: PCS::Commitment,
    pub h_comm: PCS::Commitment,
}

pub trait LookupCheck<E, PCS>: ZeroCheck<E::ScalarField>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<E>,
{
    type LookupCheckProof;
    type LookupCheckSubClaim;

    /// Initialize the system with a transcript
    ///
    /// This function is optional -- in the case where a LookupCheck is
    /// an building block for a more complex protocol, the transcript
    /// may be initialized by this complex protocol, and passed to the
    /// LookupCheck prover/verifier.
    fn init_transcript() -> Self::Transcript;

    fn prove(
        pcs_param: &PCS::ProverParam,
        fxs: &[Self::MultilinearExtension],
        txs: &[Self::MultilinearExtension],
        transcript: &mut Self::Transcript,
    ) -> Result<Self::LookupCheckProof, PolyIOPErrors>;

    fn verify(
        proof: &Self::LookupCheckProof,
        aux_info: &Self::VPAuxInfo,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::LookupCheckSubClaim, PolyIOPErrors>;
}

impl<E, PCS> LookupCheck<E, PCS> for PolyIOP<E::ScalarField>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<E, Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>>,
{
    type LookupCheckSubClaim = LookupCheckSubClaim<E::ScalarField, Self>;
    type LookupCheckProof = LookupCheckProof<E, PCS, Self>;

    fn init_transcript() -> Self::Transcript {
        IOPTranscript::<E::ScalarField>::new(b"Initializing LookupCheck transcript")
    }

    fn prove(
        pcs_param: &PCS::ProverParam,
        fxs: &[Self::MultilinearExtension],
        txs: &[Self::MultilinearExtension],
        transcript: &mut Self::Transcript,
    ) -> Result<Self::LookupCheckProof, PolyIOPErrors> {
        if fxs.is_empty() {
            return Err(PolyIOPErrors::InvalidParameters("fxs is empty".to_string()));
        }
        if fxs.len() != txs.len() {
            return Err(PolyIOPErrors::InvalidParameters(
                "fxs and txs have different number of polynomials".to_string(),
            ));
        }
        let num_vars = fxs[0].num_vars;
        for poly in fxs.iter().chain(txs.iter()) {
            if poly.num_vars != num_vars {
                return Err(PolyIOPErrors::InvalidParameters(
                    "fx and tx have different number of variables".to_string(),
                ));
            }
        }

        let theta = transcript.get_and_append_challenge(b"theta")?;

        // fx = f1 + f2*theta + f3*theta^2 + ...
        // tx = t1 + t2*theta + t3*theta^2 + ...
        let [fx, tx] = [fxs, txs].map(|polys| {
            let poly = polys
                .iter()
                .zip(iter::successors(
                    Some(E::ScalarField::one()),
                    |power_of_theta| Some(theta * power_of_theta),
                ))
                .fold(
                    DenseMultilinearExtension::zero(),
                    |mut acc, (fx, power_of_theta)| {
                        acc += (power_of_theta, fx.deref());
                        acc
                    },
                );
            Arc::new(poly)
        });

        // Count how many times each element in table is used.
        let m_poly = {
            let map = tx.iter().zip(0..).collect::<HashMap<_, _>>();

            let mut m = vec![E::ScalarField::zero(); 1 << num_vars];
            for f in fx.iter() {
                if let Some(idx) = map.get(f) {
                    m[*idx] += E::ScalarField::one();
                } else {
                    return Err(PolyIOPErrors::InvalidProver(
                        "Invalid input in lookup".to_string(),
                    ));
                }
            }

            Arc::new(DenseMultilinearExtension::from_evaluations_vec(num_vars, m))
        };

        let m_comm = PCS::commit(pcs_param, &m_poly)?;
        transcript.append_serializable_element(b"m(x)", &m_comm)?;

        let tau = transcript.get_and_append_challenge(b"tau")?;

        // f_prime = fx + tau
        // t_prime = tx + tau
        // h = 1/f_prime - m/t_prime
        let [f_prime_poly, t_prime_poly, h_poly] = {
            let mut f_prime = vec![E::ScalarField::zero(); 1 << num_vars];
            let mut t_prime = vec![E::ScalarField::zero(); 1 << num_vars];

            f_prime
                .par_iter_mut()
                .zip(fx.evaluations.par_iter())
                .for_each(|(f_prime, fx)| {
                    *f_prime = tau + fx;
                });
            t_prime
                .par_iter_mut()
                .zip(tx.evaluations.par_iter())
                .for_each(|(t_prime, tx)| {
                    *t_prime = tau + tx;
                });
            let mut f_inv = f_prime.clone();
            let mut t_inv = t_prime.clone();
            batch_inversion(&mut f_inv);
            batch_inversion(&mut t_inv);

            let h = f_inv
                .par_iter()
                .zip(t_inv.par_iter())
                .zip(m_poly.evaluations.par_iter())
                .map(|((f_inv, t_inv), m)| *f_inv - *t_inv * m)
                .collect::<Vec<_>>();

            [f_prime, t_prime, h].map(|poly| {
                Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                    num_vars, poly,
                ))
            })
        };

        let h_comm = PCS::commit(pcs_param, &m_poly)?;
        transcript.append_serializable_element(b"h(x)", &m_comm)?;

        // Challenge for batch zero check
        let alpha = transcript.get_and_append_challenge(b"alpha")?;

        let r = transcript.get_and_append_challenge_vectors(b"0check r", num_vars)?;

        // Build VirtualPolynomial for zero check the following relation:
        // - eq(X,r)*(h(X)*f'(X)*t'(X) - t'(X) + m(X)*f'(X))
        // - h(X)
        // with alpha as batching challenge.
        let mut f = VirtualPolynomial::new(fx.num_vars);
        f.add_mle_list(
            [h_poly.clone(), f_prime_poly.clone(), t_prime_poly.clone()],
            E::ScalarField::one(),
        )?;
        f.add_mle_list([t_prime_poly], -E::ScalarField::one())?;
        f.add_mle_list([m_poly, f_prime_poly], E::ScalarField::one())?;
        let mut f_hat = f.build_f_hat(&r)?;
        f_hat.add_mle_list([h_poly], alpha)?;

        let zero_check_proof = <Self as SumCheck<E::ScalarField>>::prove(&f_hat, transcript)?;

        Ok(LookupCheckProof {
            zero_check_proof,
            m_comm,
            h_comm,
        })
    }

    fn verify(
        proof: &Self::LookupCheckProof,
        aux_info: &Self::VPAuxInfo,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::LookupCheckSubClaim, PolyIOPErrors> {
        let theta = transcript.get_and_append_challenge(b"theta")?;

        transcript.append_serializable_element(b"m(x)", &proof.m_comm)?;

        let tau = transcript.get_and_append_challenge(b"tau")?;

        transcript.append_serializable_element(b"h(x)", &proof.h_comm)?;

        let alpha = transcript.get_and_append_challenge(b"alpha")?;

        let zero_check_sub_claim = <Self as ZeroCheck<E::ScalarField>>::verify(
            &proof.zero_check_proof,
            aux_info,
            transcript,
        )?;

        Ok(LookupCheckSubClaim {
            zero_check_sub_claim,
            theta,
            tau,
            alpha,
        })
    }
}

#[cfg(test)]
mod test {
    use crate::{
        poly_iop::lookup_check::LookupCheck, MultilinearKzgPCS, PolyIOP, PolyIOPErrors,
        PolynomialCommitmentScheme,
    };
    use arithmetic::VPAuxInfo;
    use ark_bls12_381::{Bls12_381, Fr};
    use ark_ec::pairing::Pairing;
    use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
    use ark_std::{rand::Rng, test_rng};
    use std::{marker::PhantomData, sync::Arc};

    fn test_lookup_check_helper<E, PCS>(
        fs: &[Arc<DenseMultilinearExtension<E::ScalarField>>],
        ts: &[Arc<DenseMultilinearExtension<E::ScalarField>>],
        pcs_param: &PCS::ProverParam,
    ) -> Result<(), PolyIOPErrors>
    where
        E: Pairing,
        PCS: PolynomialCommitmentScheme<
            E,
            Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>,
        >,
    {
        let mut transcript = <PolyIOP<E::ScalarField> as LookupCheck<E, PCS>>::init_transcript();
        transcript.append_message(b"testing", b"initializing transcript for testing")?;

        let proof = <PolyIOP<E::ScalarField> as LookupCheck<E, PCS>>::prove(
            pcs_param,
            fs,
            ts,
            &mut transcript,
        )?;

        let mut transcript = <PolyIOP<E::ScalarField> as LookupCheck<E, PCS>>::init_transcript();
        transcript.append_message(b"testing", b"initializing transcript for testing")?;

        let aux_info = VPAuxInfo {
            max_degree: 3,
            num_variables: fs[0].num_vars,
            phantom: PhantomData::default(),
        };
        let _lookup_subclaim = <PolyIOP<E::ScalarField> as LookupCheck<E, PCS>>::verify(
            &proof,
            &aux_info,
            &mut transcript,
        )?;

        Ok(())
    }

    fn test_lookup_check(nv: usize) -> Result<(), PolyIOPErrors> {
        let mut rng = test_rng();

        let t1: DenseMultilinearExtension<Fr> = DenseMultilinearExtension::rand(nv, &mut rng);
        let t2: DenseMultilinearExtension<Fr> = DenseMultilinearExtension::rand(nv, &mut rng);
        let ts = vec![Arc::new(t2), Arc::new(t1)];

        let indices: Vec<_> = (0..1 << nv).map(|_| rng.gen_range(0..1 << nv)).collect();
        let fs: Vec<_> = ts
            .iter()
            .map(|t| {
                Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                    nv,
                    indices.iter().map(|idx| t[*idx]).collect(),
                ))
            })
            .collect();

        let srs = MultilinearKzgPCS::<Bls12_381>::gen_srs_for_testing(&mut rng, nv)?;
        let (pcs_param, _) = MultilinearKzgPCS::<Bls12_381>::trim(&srs, None, Some(nv))?;

        test_lookup_check_helper::<Bls12_381, MultilinearKzgPCS<Bls12_381>>(&fs, &ts, &pcs_param)?;

        Ok(())
    }

    #[test]
    fn test_trivial_polynomial() -> Result<(), PolyIOPErrors> {
        test_lookup_check(1)
    }
    #[test]
    fn test_normal_polynomial() -> Result<(), PolyIOPErrors> {
        test_lookup_check(10)
    }
}

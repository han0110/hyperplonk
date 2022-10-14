//! Sumcheck based batch opening and verify commitment.
// TODO: refactoring this code to somewhere else
// currently IOP depends on PCS because perm check requires commitment.
// The sumcheck based batch opening therefore cannot stay in the PCS repo --
// which creates a cyclic dependency.

use crate::{
    pcs::{
        multilinear_kzg::util::eq_eval,
        prelude::{Commitment, PCSError},
        PolynomialCommitmentScheme,
    },
    poly_iop::{prelude::SumCheck, PolyIOP},
    IOPProof,
};
use arithmetic::{
    build_eq_x_r_vec, fix_last_variables, DenseMultilinearExtension, VPAuxInfo, VirtualPolynomial,
};
use ark_ec::{AffineCurve, PairingEngine, ProjectiveCurve};
use ark_std::{end_timer, log2, start_timer, One, Zero};
use std::{marker::PhantomData, rc::Rc};
use transcript::IOPTranscript;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BatchProof<E, PCS>
where
    E: PairingEngine,
    PCS: PolynomialCommitmentScheme<E>,
{
    /// A sum check proof proving tilde g's sum
    pub(crate) sum_check_proof: IOPProof<E::Fr>,
    /// f_i(point_i)
    pub f_i_eval_at_point_i: Vec<E::Fr>,
    /// proof for g'(a_2)
    pub(crate) g_prime_proof: PCS::Proof,
}

/// Steps:
/// 1. get challenge point t from transcript
/// 2. build eq(t,i) for i in [0..k]
/// 3. build \tilde g(i, b) = eq(t, i) * f_i(b)
/// 4. compute \tilde eq
/// 5. run sumcheck on \tilde eq * \tilde g(i, b)
/// 6. build g'(a2) where (a1, a2) is the sumcheck's point
pub(crate) fn multi_open_internal<E, PCS>(
    prover_param: &PCS::ProverParam,
    polynomials: &[PCS::Polynomial],
    points: &[PCS::Point],
    evals: &[PCS::Evaluation],
    transcript: &mut IOPTranscript<E::Fr>,
) -> Result<BatchProof<E, PCS>, PCSError>
where
    E: PairingEngine,
    PCS: PolynomialCommitmentScheme<
        E,
        Polynomial = Rc<DenseMultilinearExtension<E::Fr>>,
        Point = Vec<E::Fr>,
        Evaluation = E::Fr,
    >,
{
    let open_timer = start_timer!(|| format!("multi open {} points", points.len()));

    // TODO: sanity checks
    let num_var = polynomials[0].num_vars;
    let k = polynomials.len();
    let ell = log2(k) as usize;
    let merged_num_var = num_var + ell;

    // challenge point t
    let t = transcript.get_and_append_challenge_vectors("t".as_ref(), ell)?;

    // eq(t, i) for i in [0..k]
    let eq_t_i_list = build_eq_x_r_vec(t.as_ref())?;

    // \tilde g(i, b) = eq(t, i) * f_i(b)
    let timer = start_timer!(|| format!("compute tilde g for {} points", points.len()));
    let mut tilde_g_eval = vec![E::Fr::zero(); 1 << (ell + num_var)];
    let block_size = 1 << num_var;
    for (index, f_i) in polynomials.iter().enumerate() {
        for (j, &f_i_eval) in f_i.iter().enumerate() {
            tilde_g_eval[index * block_size + j] = f_i_eval * eq_t_i_list[index];
        }
    }
    let tilde_g = Rc::new(DenseMultilinearExtension::from_evaluations_vec(
        merged_num_var,
        tilde_g_eval,
    ));
    end_timer!(timer);

    let timer = start_timer!(|| format!("compute tilde eq for {} points", points.len()));
    let mut tilde_eq_eval = vec![E::Fr::zero(); 1 << (ell + num_var)];
    for (index, point) in points.iter().enumerate() {
        let eq_b_zi = build_eq_x_r_vec(point)?;
        let start = index * block_size;
        tilde_eq_eval[start..start + block_size].copy_from_slice(eq_b_zi.as_slice());
    }
    let tilde_eq = Rc::new(DenseMultilinearExtension::from_evaluations_vec(
        merged_num_var,
        tilde_eq_eval,
    ));
    end_timer!(timer);

    // built the virtual polynomial for SumCheck
    let timer = start_timer!(|| format!("sum check prove of {} variables", num_var + ell));

    let step = start_timer!(|| "add mle");
    let mut sum_check_vp = VirtualPolynomial::new(num_var + ell);
    sum_check_vp.add_mle_list([tilde_g.clone(), tilde_eq], E::Fr::one())?;
    end_timer!(step);

    let proof = match <PolyIOP<E::Fr> as SumCheck<E::Fr>>::prove(&sum_check_vp, transcript) {
        Ok(p) => p,
        Err(_e) => {
            // cannot wrap IOPError with PCSError due to cyclic dependency
            return Err(PCSError::InvalidProver(
                "Sumcheck in batch proving Failed".to_string(),
            ));
        },
    };

    end_timer!(timer);

    // (a1, a2) := sumcheck's point
    let step = start_timer!(|| "open at a2");
    let a1 = &proof.point[num_var..];
    let a2 = &proof.point[..num_var];
    end_timer!(step);

    // build g'(a2)
    let step = start_timer!(|| "evaluate at a2");
    let g_prime = Rc::new(fix_last_variables(&tilde_g, a1));
    end_timer!(step);

    let step = start_timer!(|| "pcs open");
    let (g_prime_proof, _g_prime_eval) = PCS::open(prover_param, &g_prime, a2.to_vec().as_ref())?;
    // assert_eq!(g_prime_eval, tilde_g_eval);
    end_timer!(step);

    let step = start_timer!(|| "evaluate fi(pi)");
    end_timer!(step);
    end_timer!(open_timer);

    Ok(BatchProof {
        sum_check_proof: proof,
        f_i_eval_at_point_i: evals.to_vec(),
        g_prime_proof,
    })
}

/// Steps:
/// 1. get challenge point t from transcript
/// 2. build g' commitment
/// 3. ensure \sum_i eq(t, <i>) * f_i_evals matches the sum via SumCheck
/// verification 4. verify commitment
pub(crate) fn batch_verify_internal<E, PCS>(
    verifier_param: &PCS::VerifierParam,
    f_i_commitments: &[Commitment<E>],
    points: &[PCS::Point],
    proof: &BatchProof<E, PCS>,
    transcript: &mut IOPTranscript<E::Fr>,
) -> Result<bool, PCSError>
where
    E: PairingEngine,
    PCS: PolynomialCommitmentScheme<
        E,
        Polynomial = Rc<DenseMultilinearExtension<E::Fr>>,
        Point = Vec<E::Fr>,
        Evaluation = E::Fr,
        Commitment = Commitment<E>,
    >,
{
    let open_timer = start_timer!(|| "batch verification");

    // TODO: sanity checks

    let k = f_i_commitments.len();
    let ell = log2(k) as usize;
    let num_var = proof.sum_check_proof.point.len() - ell;

    // challenge point t
    let t = transcript.get_and_append_challenge_vectors("t".as_ref(), ell)?;

    // sum check point (a1, a2)
    let a1 = &proof.sum_check_proof.point[num_var..];
    let a2 = &proof.sum_check_proof.point[..num_var];

    // build g' commitment
    let eq_a1_list = build_eq_x_r_vec(a1)?;
    let eq_t_list = build_eq_x_r_vec(t.as_ref())?;

    let mut g_prime_commit = E::G1Affine::zero().into_projective();
    for i in 0..k {
        let tmp = eq_a1_list[i] * eq_t_list[i];
        g_prime_commit += &f_i_commitments[i].0.mul(tmp);
    }

    // ensure \sum_i eq(t, <i>) * f_i_evals matches the sum via SumCheck
    // verification
    let mut sum = E::Fr::zero();
    for (i, &e) in eq_t_list.iter().enumerate().take(k) {
        sum += e * proof.f_i_eval_at_point_i[i];
    }
    let aux_info = VPAuxInfo {
        max_degree: 2,
        num_variables: num_var + ell,
        phantom: PhantomData,
    };
    let subclaim = match <PolyIOP<E::Fr> as SumCheck<E::Fr>>::verify(
        sum,
        &proof.sum_check_proof,
        &aux_info,
        transcript,
    ) {
        Ok(p) => p,
        Err(_e) => {
            // cannot wrap IOPError with PCSError due to cyclic dependency
            return Err(PCSError::InvalidProver(
                "Sumcheck in batch verification failed".to_string(),
            ));
        },
    };
    let mut eq_tilde_eval = E::Fr::zero();
    for (point, &coef) in points.iter().zip(eq_a1_list.iter()) {
        eq_tilde_eval += coef * eq_eval(a2, point)?;
    }
    let tilde_g_eval = subclaim.expected_evaluation / eq_tilde_eval;

    // verify commitment
    let res = PCS::verify(
        verifier_param,
        &Commitment(g_prime_commit.into_affine()),
        a2.to_vec().as_ref(),
        &tilde_g_eval,
        &proof.g_prime_proof,
    )?;

    end_timer!(open_timer);
    Ok(res)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pcs::{
        prelude::{MultilinearKzgPCS, MultilinearUniversalParams},
        StructuredReferenceString,
    };
    use arithmetic::get_batched_nv;
    use ark_bls12_381::Bls12_381 as E;
    use ark_ec::PairingEngine;
    use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
    use ark_std::{
        rand::{CryptoRng, RngCore},
        test_rng,
        vec::Vec,
        UniformRand,
    };

    type Fr = <E as PairingEngine>::Fr;

    fn test_multi_open_helper<R: RngCore + CryptoRng>(
        ml_params: &MultilinearUniversalParams<E>,
        polys: &[Rc<DenseMultilinearExtension<Fr>>],
        rng: &mut R,
    ) -> Result<(), PCSError> {
        let merged_nv = get_batched_nv(polys[0].num_vars(), polys.len());
        let (ml_ck, ml_vk) = ml_params.trim(merged_nv)?;

        let mut points = Vec::new();
        for poly in polys.iter() {
            let point = (0..poly.num_vars())
                .map(|_| Fr::rand(rng))
                .collect::<Vec<Fr>>();
            points.push(point);
        }

        let evals = polys
            .iter()
            .zip(points.iter())
            .map(|(f, p)| f.evaluate(p).unwrap())
            .collect::<Vec<_>>();

        let commitments = polys
            .iter()
            .map(|poly| MultilinearKzgPCS::commit(&ml_ck.clone(), poly).unwrap())
            .collect::<Vec<_>>();

        let mut transcript = IOPTranscript::new("test transcript".as_ref());
        transcript.append_field_element("init".as_ref(), &Fr::zero())?;

        let batch_proof = multi_open_internal::<E, MultilinearKzgPCS<E>>(
            &ml_ck,
            polys,
            &points,
            &evals,
            &mut transcript,
        )?;

        // good path
        let mut transcript = IOPTranscript::new("test transcript".as_ref());
        transcript.append_field_element("init".as_ref(), &Fr::zero())?;
        assert!(batch_verify_internal::<E, MultilinearKzgPCS<E>>(
            &ml_vk,
            &commitments,
            &points,
            &batch_proof,
            &mut transcript
        )?);

        Ok(())
    }

    #[test]
    fn test_multi_open_internal() -> Result<(), PCSError> {
        let mut rng = test_rng();

        let ml_params = MultilinearUniversalParams::<E>::gen_srs_for_testing(&mut rng, 20)?;
        for num_poly in 5..6 {
            for nv in 15..16 {
                let polys1: Vec<_> = (0..num_poly)
                    .map(|_| Rc::new(DenseMultilinearExtension::rand(nv, &mut rng)))
                    .collect();
                test_multi_open_helper(&ml_params, &polys1, &mut rng)?;
            }
        }

        Ok(())
    }
}
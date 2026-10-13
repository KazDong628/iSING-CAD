import numpy as np

from contour_agent.feasible_refinement import refine_feasible_point


def test_image_refinement_improves_unknowns_with_dependent_exact_rows():
    accepted=[]
    # A curved equality manifold and one redundant equality, analogous to
    # dependent CAD axis/contact rows. Start close enough to exercise the
    # bounded normalized step without a long numerical optimization.
    def eq(p):return np.asarray([p @ p-.01**2, 2*(p @ p-.01**2)])
    def jac(p):return np.asarray([2*p,4*p])
    def residual(p):return p-[.01,0.]
    def certify(p):
        if np.max(np.abs(eq(p)))>1e-10 or p[0]<0:return False
        accepted.append(p.copy());return True
    before=np.asarray([0.,.01])
    after,receipt=refine_feasible_point(before,residual,eq,jac,lambda p:np.eye(2),
        [lambda p:np.asarray([p[0]])],[-.02,-.02],[.02,.02],certify,seconds=2.)
    assert accepted and receipt['certified_improvements']>0
    assert np.linalg.norm(residual(after)) < np.linalg.norm(residual(before))*.5
    assert max(np.max(np.abs(eq(p))) for p in accepted)<1e-10
    assert not receipt['known_equations_changed'] and not receipt['publication_approved']


def test_uncertified_improvements_never_replace_initial_checkpoint():
    point=np.asarray([0.,.01])
    after,receipt=refine_feasible_point(point,lambda p:p-[.01,0.],
        lambda p:np.asarray([p @ p-.01**2]),lambda p:np.asarray([2*p]),
        lambda p:np.eye(2),[],[-.02,-.02],[.02,.02],lambda p:False,seconds=2.)
    assert np.array_equal(after,point) and receipt['certified_improvements']==0

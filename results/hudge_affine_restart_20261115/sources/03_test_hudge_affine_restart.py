"""Composition must preserve physical sampling and affine derivatives."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("restart", Path(__file__).resolve().parents[1]/"hudge_affine_restart.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_composition_sampling_and_round_trip():
    centres = np.array([[31.5,47.5],[111.5,79.5]])
    matrix = np.array([[1.2,.4],[-.3,3.1]])
    translation = np.array([74.,-25.])
    residual = np.array([[2.,.02,.03,-3.,-.05,.07],[-4.,-.03,.01,8.,.04,.02]])
    result = module.compose(centres,residual,matrix,translation)
    np.testing.assert_allclose(module.uncompose(centres,result,matrix,translation),residual,atol=1e-12)
    for x,r,p in zip(centres,residual,result,strict=True):
        def expected(offset, x=x, r=r):
            return matrix @ (x+r[[0,3]]+(np.eye(2)+r[[1,2,4,5]].reshape(2,2)) @ offset)+translation
        gradient = (np.eye(2)+p[[1,2,4,5]].reshape(2,2))
        for offset in [np.array([0.,0.]),np.array([-15.5,15.5])]:
            np.testing.assert_allclose(x+p[[0,3]]+gradient @ offset,expected(offset),atol=1e-12)
        finite_difference = np.column_stack([(expected(np.eye(2)[i]*1e-4)-expected(-np.eye(2)[i]*1e-4))/2e-4 for i in range(2)])
        np.testing.assert_allclose(gradient,finite_difference,atol=1e-8)


def test_fit_uses_accepted_points_and_rejects_collinearity():
    points = np.array([[0.,0.],[10.,0.],[0.,10.],[10.,10.],[99.,99.]])
    matrix = np.array([[1.,.2],[.1,4.]])
    translation = np.array([7.,-3.])
    states = module.affine_states(points,matrix,translation)
    states[-1] = 1e6
    a,t,_=module.fit_global(points,states,np.array([1,1,1,1,0],bool))
    np.testing.assert_allclose(a,matrix,atol=1e-12)
    np.testing.assert_allclose(t,translation,atol=1e-12)
    with pytest.raises(ValueError,match="collinear"):
        module.fit_global(np.array([[0.,0.],[1.,0.],[2.,0.]]),np.zeros((3,6)),np.ones(3,bool))


def test_rectification_samples_target_not_inverse_map():
    y,x=np.indices((40,50))
    image=(x+2*y).astype(np.uint8)
    matrix=np.array([[1.,.1],[0.,1.2]])
    translation=np.array([3.,2.])
    result=module.rectify(image,matrix,translation)
    np.testing.assert_allclose(result[5:15,5:15],((x+.1*y+3)+2*(1.2*y+2))[5:15,5:15],atol=.500001,rtol=0)

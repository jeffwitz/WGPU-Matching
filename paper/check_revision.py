"""Document validation, with optional analytic checks of existing geometry code."""
from pathlib import Path
import argparse
import json
import re
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; P=ROOT/'paper'
parser=argparse.ArgumentParser();parser.add_argument('--geometry',action='store_true');args=parser.parse_args()
if args.geometry:
    import torch
    sys.path.insert(0,str(ROOT/'python/benchmarks'))
    from ndf_field import feature_grid_points, feature_grid_displacement_field, resize_pixel_center_coordinates
    pts=feature_grid_points((192,108),(27,48));assert np.allclose(pts[0],[1.5,1.5])
    query=torch.tensor([[20.2,17.8],[90.3,40.9],[160.8,80.1]],dtype=torch.float64)
    checks=[]
    for A in [np.eye(2),np.array([[0.,-1.],[1.,0.]]),np.array([[1.6,.3],[-.12,.9]])]:
        G=torch.tensor(A-np.eye(2),dtype=torch.float64);b=torch.tensor([21.,-9.],dtype=torch.float64)
        coeff=(pts@G.T+b).reshape(27,48,2);field=feature_grid_displacement_field(coeff,(192,108))
        u,F=field.evaluate(query)
        err=float((u-(query@G.T+b)).abs().max());grad=float((F-torch.tensor(A)).abs().max())
        assert err<1e-10 and grad<1e-10
        for scale in [(20.,20.),(17.,23.)]:
            S=torch.diag(torch.tensor(scale,dtype=torch.float64));native=(query+.5)@S-.5
            assert torch.allclose(resize_pixel_center_coordinates(native,scale),query)
            nativeF=S@F@torch.linalg.inv(S)
            eps=1e-3
            for axis in range(2):
                delta=torch.zeros_like(native);delta[:,axis]=eps
                plus,_=field.evaluate(resize_pixel_center_coordinates(native+delta,scale))
                minus,_=field.evaluate(resize_pixel_center_coordinates(native-delta,scale))
                deriv=(plus-minus)@S/(2*eps)
                expected=(nativeF-torch.eye(2))[:,:,axis]
                assert torch.allclose(deriv,expected,atol=1e-8)
        checks.append({'displacement_error':err,'gradient_error':grad})
    # Bilinear non-affine function: exactly representable in each cell.
    coeff=torch.stack((.002*pts[:,0]*pts[:,1],-.001*pts[:,0]*pts[:,1]),1).reshape(27,48,2)
    u,F=feature_grid_displacement_field(coeff,(192,108)).evaluate(query)
    expected=torch.stack((.002*query[:,0]*query[:,1],-.001*query[:,0]*query[:,1]),1)
    assert torch.allclose(u,expected,atol=1e-10)
    G=torch.stack((torch.stack((.002*query[:,1],.002*query[:,0]),1),torch.stack((-.001*query[:,1],-.001*query[:,0]),1)),1)
    assert torch.allclose(F,G+torch.eye(2),atol=1e-10)
    # Unequal scale/crop/padding maps: algebraic round trip, not supported model input.
    B0=np.diag([1/20,1/18]);B1=np.diag([1/16,1/22]);b0=np.array([2.,-3.]);b1=np.array([-4.,5.]);A=np.array([[1.1,.2],[-.1,.9]]);t=np.array([7.,-8.]);x=np.array([150.,280.])
    phi=lambda x:np.linalg.solve(B1,A@(B0@x+b0)+t-b1)
    J=np.linalg.solve(B1,A@B0);eps=1e-4
    assert np.allclose(np.column_stack([(phi(x+np.eye(2)[i]*eps)-phi(x-np.eye(2)[i]*eps))/(2*eps) for i in range(2)]),J,atol=1e-8)
    # Forward preconditioning after a nonlinear residual; inverse affine increment.
    A=np.array([[1.3,.2],[-.1,.8]]);t=np.array([3.,-5.]);x=np.array([12.,17.])
    residual=lambda z:np.array([.002*z[0]*z[1],-.001*z[0]*z[1]])
    D=np.array([[.002*x[1],.002*x[0]],[-.001*x[1],-.001*x[0]]])
    composed=lambda z:A@(z+residual(z))+t
    numerical=np.column_stack([(composed(x+np.eye(2)[i]*eps)-composed(x-np.eye(2)[i]*eps))/(2*eps) for i in range(2)])
    assert np.allclose(numerical,A@(np.eye(2)+D),atol=1e-8)
    increment=np.array([[1.01,.02],[-.01,.98]]);shift=np.array([.3,-.2])
    inverse=lambda z:np.linalg.solve(increment,z-shift)
    assert np.allclose(increment@inverse(x)+shift,x)
    assert np.allclose(A@inverse(x)+t,(A@np.linalg.inv(increment))@x+t-A@np.linalg.solve(increment,shift))
    (P/'audit/geometry_checks.json').write_text(json.dumps({'affine':checks,'forward_and_inverse_composition':'pass','nonaffine_bilinear':'pass','anisotropic':'pass','unequal_maps_algebra_only':'pass','scope':'interior feature grid; no inference or solver rerun'},indent=2)+'\n')
else:
    result={}
    for name in ['main','supplement']:
        log=(P/(name+'.log')).read_text()
        for forbidden in ['Overfull', 'undefined references', 'undefined citations', 'multiply defined']:
            assert forbidden not in log,(name,forbidden)
        assert not re.search(r'^(?:LaTeX Warning: (?:Reference|Citation).+undefined|!)',log,re.M)
        result[name]='compiled without unresolved references or overfull boxes'
    text=(P/'main.tex').read_text();abstract=text.split('\\begin{abstract}')[1].split('\\end{abstract}')[0]
    result['abstract_words']=len(abstract.split());assert result['abstract_words']<=300
    data=json.loads((P/'audit/recount.json').read_text());result['causal_totals']={k:v['propagated'] for k,v in data['causal'].items()}
    result['geometry_checks']=json.loads((P/'audit/geometry_checks.json').read_text())
    (P/'audit/editorial_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

"""Exact rational diagnostic on represented coordinates, not a full raycaster."""
from fractions import Fraction


def sub(a,b):return tuple(x-y for x,y in zip(a,b))
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def cross(a,b):return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
def vector(a):
    if len(a)!=3:raise ValueError('xyz required')
    return tuple(Fraction(float(x)) for x in a)


def exact_triangle_ray(triangle,origin,direction):
    if len(triangle)!=3:raise ValueError('triangle required')
    a,b,c=map(vector,triangle);o=vector(origin);d=vector(direction)
    if not any(d):raise ValueError('nonzero direction required')
    e1=sub(b,a);e2=sub(c,a);n=cross(e1,e2);norm=dot(n,n)
    if not norm:return {'status':'degenerate'}
    denominator=dot(n,d)
    if not denominator:return {'status':'coplanar' if dot(n,sub(a,o))==0 else 'parallel'}
    t=dot(n,sub(a,o))/denominator
    p=tuple(x+t*y for x,y in zip(o,d));q=sub(p,a)
    u=dot(cross(q,e2),n)/norm;v=dot(cross(e1,q),n)/norm
    valid=t>0 and u>=0 and v>=0 and u+v<=1
    pivot=next(x for x in n if x)
    plane=tuple(x/pivot for x in n)+(dot(n,a)/pivot,)
    return {'status':'hit' if valid else 'outside_triangle','t':t,'u':u,'v':v,
            'boundary':u==0 or v==0 or u+v==1,'plane':plane}

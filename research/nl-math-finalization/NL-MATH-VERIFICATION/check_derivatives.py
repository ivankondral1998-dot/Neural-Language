"""Tiny analytic/central-difference checks, no parameter updates or training."""
import math,json,copy
def mm(a,b):return [[sum(x*y for x,y in zip(row,col)) for col in zip(*b)] for row in a]
def tr(a):return list(map(list,zip(*a)))
def inner(a,b):return sum(x*y for ar,br in zip(a,b) for x,y in zip(ar,br))
def fd(fun,x,h):
    ans=[]
    for i,row in enumerate(x):
        line=[]
        for j in range(len(row)):
            p=copy.deepcopy(x);m=copy.deepcopy(x);p[i][j]+=h;m[i][j]-=h
            line.append((fun(p)-fun(m))/(2*h))
        ans.append(line)
    return ans
def err(a,b):return max(abs(x-y) for ar,br in zip(a,b) for x,y in zip(ar,br))
A=[[.3,-.8],[.2,.6]];B=[[.5,.1],[-.4,.7]];G=[[.2,-.3],[.9,.4]]
R={}
for h in [1e-4,1e-5,1e-6]:
    R[f'matmul_A_{h}']=err(mm(G,tr(B)),fd(lambda a:inner(mm(a,B),G),A,h))
    R[f'matmul_B_{h}']=err(mm(tr(A),G),fd(lambda b:inner(mm(A,b),G),B,h))
def ln(x,g,b,eps):
    mu=sum(x)/len(x);u=[v-mu for v in x];r=math.sqrt(sum(v*v for v in u)/len(x)+eps)
    return [gg*v/r+bb for gg,v,bb in zip(g,u,b)]
x=[.3,-.8,.2];gam=[.5,1.2,-.4];bet=[.1,.2,.3];eps=1e-3;up=[.2,-.3,.9]
mu=sum(x)/3;u=[a-mu for a in x];r=math.sqrt(sum(a*a for a in u)/3+eps)
dx=[[sum(up[i]*gam[i]*((int(i==j)-1/3)/r-u[i]*u[j]/(3*r**3)) for i in range(3)) for j in range(3)]]
for h in [1e-4,1e-5,1e-6]:R[f'LN_{h}']=err(dx,fd(lambda a:sum(v*g for v,g in zip(ln(a[0],gam,bet,eps),up)),[x],h))
def att(q,k,v):
    s=mm(q,tr(k));p=[]
    for i,row in enumerate(s):
        z=[a/math.sqrt(len(q[0])) for a in row[:i+1]];mx=max(z);w=[math.exp(a-mx) for a in z];den=sum(w)
        p.append([a/den for a in w]+[0.]*(len(row)-len(w)))
    return mm(p,v),p
q=[[.3,-.8],[.2,.6]];k=[[.5,.1],[-.4,.7]];v=[[.1,.8],[-.3,.2]]
o,p=att(q,k,v);dp=mm(G,tr(v));ds=[]
for pr,dr in zip(p,dp):
    s=sum(a*b for a,b in zip(pr,dr));ds.append([a*(b-s) for a,b in zip(pr,dr)])
dq=[[x/math.sqrt(2) for x in row] for row in mm(ds,k)]
dk=[[x/math.sqrt(2) for x in row] for row in mm(tr(ds),q)]
dv=mm(tr(p),G)
for h in [1e-4,1e-5,1e-6]:
    R[f'attention_Q_{h}']=err(dq,fd(lambda a:inner(att(a,k,v)[0],G),q,h))
    R[f'attention_K_{h}']=err(dk,fd(lambda a:inner(att(q,a,v)[0],G),k,h))
    R[f'attention_V_{h}']=err(dv,fd(lambda a:inner(att(q,k,a)[0],G),v,h))
assert max(R.values())<1e-7,R
print(json.dumps({'max_abs_residuals':R,'threshold_declared':1e-7,'scope':'tiny smooth real-operator checks; no training; no global machine accuracy proof'},indent=2))

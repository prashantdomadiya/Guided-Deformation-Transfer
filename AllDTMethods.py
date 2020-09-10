

bl_info = {
    "name": "Deformation Transfer",
    "author": "Prashant Domadiya",
    "version": (1, 0),
    "blender": (2, 81, 0),
    "location": "Python Console > Console > Run Script",
    "description": "Transfer Deformation From Sorce Temporal Sequence to Target Temporal Sequence",
    "warning": "",
    "wiki_url": "",
    "tracker_url": "",
    "category": "Development"}


import sys
sys.path.append('/home/prashant/anaconda3/envs/Blender269/lib/python3.4/site-packages')

import bpy
import bmesh as bm
import numpy as np
import os
from scipy import sparse as sp
from sksparse import cholmod as chmd
from scipy.sparse.linalg import inv
from multiprocessing import Pool
from functools import partial
import time
import itertools as it


#########################################################################################
#                   Display Mesh
#########################################################################################

def CreateMesh(V,F,NPs):
    E=np.zeros(np.shape(V))
    F = [ [int(i) for i in thing] for thing in F]
    for i in range(NPs):
        E[:,3*i]=V[:,3*i]
        E[:,3*i+1]=-V[:,3*i+2]
        E[:,3*i+2]=V[:,3*i+1]
        me = bpy.data.meshes.new('MyMesh')
        ob = bpy.data.objects.new('VG'+str(i), me)
        bpy.context.collection.objects.link(ob)
        me.from_pydata(E[:,3*i:3*i+3], [], F)
        me.update()
        


def ConnectionMatrices(CrT,fcs,NV):
    A=sp.lil_matrix((6*len(CrT),3*NV))
    t=0
    i=0
    for f in fcs:
        if t in CrT:
            NVF=len(f)
            FF=np.reshape(3*np.array([f]*3)+np.array([[0],[1],[2]]),3*len(f),order='F')
            A[6*i:6*i+6,FF]=np.array([[-1,0,0,1,0,0,0,0,0],[0,-1,0,0,1,0,0,0,0],[0,0,-1,0,0,1,0,0,0],
                                      [-1,0,0,0,0,0,1,0,0],[0,-1,0,0,0,0,0,1,0],[0,0,-1,0,0,0,0,0,1.0]])
            i+=1
        t+=1
    return A

def ReadFaces(fileName):
    F=[]
    fl = open(fileName, 'r')
    for line in fl:
        words = line.split()
        l=len(words)
        tmp=[]
        for i in range(l):
            tmp.append(int(words[i]))
        F.append(tmp)
    return F

def ReadCorrespondences(fileName,TF):
    #Count=len(open(fileName).readlines())
    CrT=[]
    CrS=[]
    fl = open(fileName, 'r')
    for line in fl:
        words = line.split()
        l=len(words)
        CrT.append(int(words[1]))
        CrS.append(int(words[0]))
    return CrT, CrS

def GetFaceVrtz(SourceInpt,SF,TF,CrS,CrT):
    PoseF=[]
    t=0
    for i in range(len(TF)):
        if (i in CrT) ==True:
            f=SF[CrS[i][0]]
            FS=np.reshape(3*np.array([f]*3)+np.array([[0],[1],[2]]),3*len(f),order='F')
            PoseF.append(np.reshape(SourceInpt[FS],3*len(f)*len(SourceInpt[0]),order='F').tolist()+ [len(f)]+ [t])
            t=t+len(TF[i])
        
    return PoseF


def GetNormal(Vec):
    I=np.array(list(itr.combinations(range(len(Vec)),2)))
    N=np.cross(Vec[I[:,0],:],Vec[I[:,1],:])
    nr=np.array([0,0,0])
    if (len(N)-np.count_nonzero(np.linalg.norm(N,axis=1)))==0:
        N=N/np.array([(np.linalg.norm(N,axis=1)).tolist()]*3).T
        X=np.sum(abs(np.dot(N,Vec.T)),axis=1)
        nr=N[np.argmin(X),:]
    return nr

def save_sparse_csc(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)

def load_sparse_csc(filename):
    loader = np.load(filename)
    return sp.csc_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])

    
#############################################################################################
#                       Sumner and Popovic
#############################################################################################



def ComputeDeformation(Ftrs):
    RF=np.zeros((3,3))
    DF=np.zeros((3,3))
        
    RF[0]=Ftrs[0:3]
    RF[1]=Ftrs[3:6]
    Nr=np.cross(RF[0],RF[1])
    Nr=Nr/np.linalg.norm(Nr)
    RF[2]=Nr

    DF[0]=Ftrs[6:9]
    DF[1]=Ftrs[9:12]
    Nr=np.cross(DF[0],DF[1])
    Nr=Nr/np.linalg.norm(Nr)
    DF[2]=Nr

    Q=np.dot(DF.T,np.linalg.inv(RF.T))
    return Q

def DTSumAndPop(Src,Vref,SF,CrS):
    Vdef=np.zeros(np.shape(Vref))
    PrllIn=np.zeros((len(CrS),12))
    t=0
    for i in CrS:
        f=SF[i]
        PrllIn[t,0:3]=Src[3*f[1]:3*f[1]+3,0]-Src[3*f[0]:3*f[0]+3,0]
        PrllIn[t,3:6]=Src[3*f[2]:3*f[2]+3,0]-Src[3*f[0]:3*f[0]+3,0]
        PrllIn[t,6:9]=Src[3*f[1]:3*f[1]+3,1]-Src[3*f[0]:3*f[0]+3,1]
        PrllIn[t,9:12]=Src[3*f[2]:3*f[2]+3,1]-Src[3*f[0]:3*f[0]+3,1]
        t+=1
        
    p=Pool()
    PrllOut=p.map(ComputeDeformation,PrllIn)
    for t in range(len(CrS)):
        for i in range(2):
            Vdef[6*t+3*i:6*t+3*i+3]=PrllOut[t].dot(Vref[6*t+3*i:6*t+3*i+3])
    p.close()

    return Vdef


####################################################################################################
#                                    DT Manifold
####################################################################################################
def VecRotation(rotateTowardVec, targetVec):
    # Rotate 'targetVec' towards 'rotateTowardVec'
    w=np.cross(targetVec,rotateTowardVec)
    if np.linalg.norm(w)==0.0:
        R=np.eye(3)
    else:
        w=w/np.linalg.norm(w)
        Dot_prdct=np.dot(rotateTowardVec,targetVec)
        tmp=Dot_prdct/(np.linalg.norm(rotateTowardVec)*np.linalg.norm(targetVec))
        if tmp>1.0:
            tmp=1.0
        elif tmp<-1.0:
            tmp=-1.0
        else:
            tmp=tmp
        theta=np.arccos(tmp)
        
        S=np.sin(theta)
        C=np.cos(theta)
        T=1-C
        R=np.array([[T*(w[0]**2)+C,T*w[0]*w[1]-S*w[2], T*w[0]*w[2]+S*w[1]],[T*w[0]*w[1]+S*w[2],T*(w[1]**2)+C,T*w[1]*w[2]-S*w[0]],[T*w[0]*w[2]-S*w[1],T*w[1]*w[2]+S*w[0],T*(w[2]**2)+C]])
    return R    

def CanonicalForm(T):
    C=np.array([np.linalg.norm(T[1,:]),0,0])
    R1=VecRotation(C,T[1,:])
    X=R1.dot(T[2,:])
    c=X[1]/(np.sqrt(np.sum(X[1:]**2)))
    s=-X[2]/(np.sqrt(np.sum(X[1:]**2)))
    R2=np.array([[1,0,0],[0,c,-s],[0,s,c]])
    return np.dot(R2,R1)

def GetScaleAndTransform(T1,T2):
    
    S=abs(T2[1,0]/T1[1,0])
    if T1[2,1]==0:
        T1[2,1]=0.01
    A=np.array([[1,(T2[2,0]-S*T1[2,0])/(S*T1[2,1]),0],[0,T2[2,1]/(S*T1[2,1]),0],[0,0,1]])  
    return S*A


def GetDeformation(In):
    TS0=np.reshape(In[0:9],(3,3))
    TS1=np.reshape(In[9:18],(3,3))
    TT0=np.reshape(In[18:],(3,3))
    RS0=CanonicalForm(TS0-TS0[0,:])
    RS1=CanonicalForm(TS1-TS1[0,:])
    RT0=CanonicalForm(TT0-TT0[0,:])
    
    B=np.dot(RT0,RS0.T)
    C=np.dot(RS1.T,B.T)
    D=GetScaleAndTransform(np.dot(RS0,(TS0-TS0[0,:]).T).T,np.dot(RS1,(TS1-TS1[0,:]).T).T)
    E=np.dot(RT0,(TT0[1:]-TT0[0,:]).T)
    return np.dot(np.dot(C,D),E).T



def DTManifold(Src,Trgt,Crrt,Crrs):

    S0=np.reshape(Src[:,0],(len(Src)//3,3))
    S1=np.reshape(Src[:,1],(len(Src)//3,3))
    P0=np.reshape(Trgt,(len(Trgt)//3,3))

    NV,nop=np.shape(P0)
    NF=int(len(Crrt)/3)
    NVF=3

    b=np.zeros([2*NF,3])
    
    p=Pool()
    PrllIn=np.concatenate((np.reshape(S0[Crrs,:],(NF,9)),
                           np.reshape(S1[Crrs,:],(NF,9)),
                           np.reshape(P0[Crrt,:],(NF,9))
                           ),axis=1)
    PrllOut=p.map(GetDeformation,PrllIn)
    for t in range(NF):
        b[2*t:2*t+2,:]=PrllOut[t]
    p.close()    
    return b

def WriteAsTxt(Name,Vec):
    with open(Name, 'w') as fl:
        for i in Vec:
            if str(type(i))=="<class 'list'>":
                for j in i:
                    fl.write(" %d" % j)
                fl.write("\n")
            else:
                fl.write(" %d" % i)

def ReadTxt(Name):
    Vec=[]
    fl = open(Name, 'r')
    NumLine=0
    for line in fl:
        words = line.split()
        l=len(words)
        tmp=[]
        for i in range(l):     
            tmp.append(int(words[i]))
        Vec.append(tmp)
        NumLine+=1
    return Vec

def ComuteReducedFaceList(F):
    bpy.ops.object.mode_set(mode="EDIT")
    obj = bpy.context.active_object.data
    m=bm.from_edit_mesh(obj)
    B=set()
    for vrt in m.verts:
        X=[]
        ASF=[]
        FV=set()
        Num_link_face=0
        for fc in vrt.link_faces:
            Num_link_face+=1
            if fc.index in B:
                ASF.append(fc.index)
                FV.update(F[fc.index])
            else:
                X.append(fc.index)
        if Num_link_face>2:
            for f in X:
                for t in it.combinations(F[f],2):
                    if len(set(list(t)).intersection(FV))==0:
                        ASF.append(f)
                        FV.update(F[f])      
                        break
            B.update(ASF)
        else:
            B.update(X)
                    
    bpy.ops.object.mode_set(mode ="OBJECT")
    return B    



#####################################################################################################
#           Deformation Transfer
#####################################################################################################

class DTToolsPanel(bpy.types.Panel):
    bl_label = "DT Tools Panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
 
    def draw(self, context):
        self.layout.operator("get.seq",text='Source Seq').seqType="source"
        self.layout.operator("get.seq",text='Target Seq').seqType="target"
        self.layout.prop(context.scene,"CorrPath")
        self.layout.operator("dt.tools",text='VGDT').seqType="VGDT"
        self.layout.operator("dt.tools", text='GDT').seqType="GDT"
        self.layout.operator("dt.tools",text='DTSumnerPopovic').seqType="DTSumnerPopovic"
        self.layout.operator("dt.tools",text='DTManifold').seqType="DTManifold" 
        

# Operator

class GetSequence(bpy.types.Operator):
    bl_idname = "get.seq"
    bl_label = "Target Sequence"
    seqType : bpy.props.StringProperty()
 
    def execute(self, context):
        path='/home/student/Documents/codeFiles/'
        
        Selected_Meshes=bpy.context.selected_objects
        obj = bpy.context.active_object
        if self.seqType=='target':
            F=[]
            for f in obj.data.polygons:
                F.append(list(f.vertices))            
            FcList=ComuteReducedFaceList(F)
            
            #FcList=[i for i in range(len(F))]
            WriteAsTxt(path+self.seqType+'_SelFcz.txt',FcList)
        
        
        V=np.zeros([3*len(obj.data.vertices),len(Selected_Meshes)])
        NPs=len(Selected_Meshes)
        for i in range(len(Selected_Meshes)):
            bpy.context.view_layer.objects.active = Selected_Meshes[i]
            obj = bpy.context.active_object
             
            t=0
            for v in obj.data.vertices:
                co_final= obj.matrix_world @ v.co
                V[3*t:3*t+3,i]=np.array([co_final.x,co_final.z,-co_final.y])
                t+=1

        bpy.context.view_layer.objects.active = Selected_Meshes[0]
        filepath=path+self.seqType+'_facez.txt'  
        if len(obj.data.polygons)!=0: 
            with open(filepath, 'w') as fl:
                for f in obj.data.polygons:
                    for i in f.vertices[:]:
                        fl.write(" %d" % i)
                    fl.write("\n")
        else:
            with open(filepath, 'w') as fl:
                for e in obj.data.edges:
                    for i in e.vertices[:]:
                        fl.write(" %d" % i)
                    fl.write("\n")
            
        np.savetxt(path+self.seqType+'_vertz.txt',V,delimiter=',')                   
        return{'FINISHED'} 

class DeformationTransferTools(bpy.types.Operator):
    bl_idname = "dt.tools"
    bl_label = "DT Tools"
    seqType : bpy.props.StringProperty()
    def execute(self,context):
        path='/home/student/Documents/codeFiles/'#bpy.utils.resource_path('USER') 
        
        SourceInpt=np.loadtxt(path+'source_vertz.txt',delimiter=',')
        TrgtInpt=np.loadtxt(path+'target_vertz.txt',delimiter=',')
        
        
        if int(np.size(TrgtInpt)/len(TrgtInpt))==1:
            TrgtInpt=np.reshape(TrgtInpt,(len(TrgtInpt),1))

        SF=ReadFaces(path+'source_facez.txt')
        TF=ReadFaces(path+'target_facez.txt')
        
        if context.scene.CorrPath!='Default':
            CrT,CrS=ReadCorrespondences(context.scene.CorrPath,TF)
        else:
            CrT=list(range(len(TF)))
            CrS=[i for i in CrT]
        NVrt=len(TrgtInpt)//3
        
        
        NPs=len(SourceInpt[0,:])
        if self.seqType=='DTSumnerPopovic':
            strttime=time.time()
            X=ConnectionMatrices(CrT,TF,NVrt)
            factor=chmd.cholesky(((X[:,:3*(NVrt-1)].transpose().dot(X[:,:3*(NVrt-1)]))).tocsc())
            Vref=X.dot(TrgtInpt[:,0])
            print("Preprocessing....",time.time()-strttime)

            strttime=time.time()
            Vdef=DTSumAndPop(SourceInpt[:,0:2],Vref,SF,CrS)
            Pose=factor(X[:,:3*(NVrt-1)].transpose().dot(Vdef))
            Pose=np.append(Pose,np.zeros(3))
            print("Pose Computation .....",time.time()-strttime)
            
            CreateMesh(np.reshape(Pose,(NVrt,3)),TF,1)
        else:
            strttime=time.time()
            A=sp.lil_matrix((2*len(CrT),NVrt))
            i=0
            for t in CrT:
                A[2*i:2*i+2,TF[t]]=np.array([[-1,1,0],[-1,0,1]])
                i+=1
            A=A.tocsc()
            factor=chmd.cholesky_AAt((A[:,0:(NVrt-1)].transpose()).tocsc())
            print("Preprocessing....",time.time()-strttime)

            strttime=time.time()
            Crrs=[]
            Crrt=[]
            for i in range(len(CrT)):
                Crrt=Crrt+TF[CrT[i]]
                Crrs=Crrs+SF[CrS[i]]
            Y=DTManifold(SourceInpt[:,0:2],TrgtInpt[:,0],Crrt,Crrs)
            P1=np.zeros([NVrt,3])
            P1[:NVrt-1]=factor(A[:,0:NVrt-1].transpose().dot(Y))
            print("Pose Computation .....",time.time()-strttime)
            
            CreateMesh(P1,TF,1)
        
        return {'FINISHED'}

def register():
    bpy.utils.register_class(DTToolsPanel)
    bpy.utils.register_class(GetSequence)
    bpy.types.Scene.CorrPath=bpy.props.StringProperty(name="Correspondence", description="Face Correspondences", default="Default")
    bpy.utils.register_class(DeformationTransferTools)
    
   

def unregister():
    bpy.utils.unregister_class(DTToolsPanel)
    bpy.utils.unregister_class(GetSequence)
    del bpy.types.Scene.CorrPath
    bpy.utils.unregister_class(DeformationTransferTools)
 
if __name__ == "__main__":  # only for live edit.
    bpy.utils.register_module(__name__) 


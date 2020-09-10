
bl_info = {
    "name": "Deformation Transfer using PI",
    "author": "Prashant Domadiya",
    "version": (1, 0),
    "blender": (2, 67, 0),
    "location": "Python Console > Console > Run Script",
    "description": "Transfer Deformation From Sorce Temporal Sequence to Target Temporal Sequence",
    "warning": "",
    "wiki_url": "",
    "tracker_url": "",
    "category": "Development"}

import bpy
import numpy as np
import os
from scipy import sparse as sp
from scipy.sparse import linalg as sl
from multiprocessing import Pool
from functools import partial
import time

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
        ob = bpy.data.objects.new('Myobj', me)
        scn = bpy.context.scene
        scn.objects.link(ob)
        me.from_pydata(E[:,3*i:3*i+3], [], F)
        me.update()

#########################################################################################
#                   Vector Rotation
#########################################################################################

def VecRotation(rotateTowardVec, targetVec):
    # Rotate 'targetVec' towards 'rotateTowardVec'
    w=np.cross(targetVec,rotateTowardVec)
    if np.linalg.norm(w)==0.0:
        R=np.eye(3)
        theta=0
    else:
        w=w/np.linalg.norm(w)
        Dot_prdct=np.dot(rotateTowardVec,targetVec)
        tmp=Dot_prdct/(np.linalg.norm(rotateTowardVec)*np.linalg.norm(targetVec))
        if tmp>1.0:
            theta=0.0
        else:
            theta=np.arccos(tmp)
        
        S=np.sin(theta)
        C=np.cos(theta)
        T=1-C
        R=np.array([[T*(w[0]**2)+C,T*w[0]*w[1]-S*w[2], T*w[0]*w[2]+S*w[1]],[T*w[0]*w[1]+S*w[2],T*(w[1]**2)+C,T*w[1]*w[2]-S*w[0]],[T*w[0]*w[2]-S*w[1],T*w[1]*w[2]+S*w[0],T*(w[2]**2)+C]])
    return R    

#########################################################################################
#                   Face Transformation
#########################################################################################
 
def FaceTransform(TargateTri,SourceTri,NVF):
    # Function gives transformation "T" which transforms "SourceTri" to "TargateTri"
    T=np.zeros([3*NVF,3*NVF])
    if NVF!=1:
        TargateTri=TargateTri-np.mean(TargateTri,axis=0)
        SourceTri=SourceTri-np.mean(SourceTri,axis=0)
    for i in range(NVF):
        R=VecRotation(TargateTri[i,:],SourceTri[i,:])
        V=np.dot(R,SourceTri[i,:].T)
        A=np.linalg.norm(TargateTri[i,:])/np.linalg.norm(V)
        R=A*R
        T[3*i:3*i+3,3*i:3*i+3]=R     
    return T

#############################################################################################
#                       PI Close Form
#############################################################################################
def GetMatrices(In):
    Q=np.reshape(In[0:len(In)/2],(len(In)/6,3))
    T=np.reshape(In[len(In)/2:],(len(In)/6,3))
    temp=FaceTransform(T-np.mean(T,axis=0),Q-np.mean(Q,axis=0),len(Q))
    return np.reshape(temp,np.size(temp))

def ConnectionMatrices(SRC,TRGT,fcs,NV,NF,NVF):
    CC=time.time()
    Phi=sp.lil_matrix((3*NVF*NF,3*NVF*NF))
    B=sp.lil_matrix((3*NVF*NF,3*NV))
    A=sp.lil_matrix((3*NVF*NF,3*NV))

    f=np.reshape(fcs,NVF*NF)
    SRCIni=np.reshape(SRC[:,0],(NV,3))
    SRCIni=np.reshape(SRCIni[f,:],(NF,3*NVF))
    TRGTIni=np.reshape(TRGT[f,:],(NF,3*NVF))
    InPut=np.concatenate((SRCIni,TRGTIni),axis=1)

    p=Pool()
    Y=p.map(GetMatrices,InPut)

    FF=np.concatenate((3*np.reshape(fcs[:,0],(NF,1)),3*np.reshape(fcs[:,0],(NF,1))+1,3*np.reshape(fcs[:,0],(NF,1))+2),axis=1)
    for i in range(1,NVF):
        FF=np.concatenate((FF,3*np.reshape(fcs[:,i],(NF,1)),3*np.reshape(fcs[:,i],(NF,1))+1,3*np.reshape(fcs[:,i],(NF,1))+2),axis=1)
    
    for t in range(NF):
        Phi[3*NVF*t:3*NVF*t+3*NVF,3*NVF*t:3*NVF*t+3*NVF]=np.reshape(np.array(Y[t]),(3*NVF,3*NVF))
        B[3*NVF*t:3*NVF*t+3*NVF,FF[t,:]]=(np.array([[1,0,0]*NVF,[0,1,0]*NVF,[0,0,1]*NVF]*NVF,dtype=float))/NVF
        A[3*NVF*t:3*NVF*t+3*NVF,FF[t,:]]=np.eye(3*NVF)
    p.close()
    print("Preprocessing ....", time.time()-CC)
    #InvMltply=sp.diags((1.0/A.sum(0)).tolist()[0],0)
    #PsdA=(A.dot(InvMltply)).transpose()
    #X=PsdA.dot(Phi.dot(A-B)+B)


    CC=time.time()
    Y=Phi.dot((A-B).dot(SRC))
    tmp=(A-B).tocsc()
    I=(tmp[:,:3*(NV-1)].transpose()).dot(tmp[:,:3*(NV-1)])
    Y=Y-tmp[:,3*(NV-1):].dot(SRC[3*(NV-1):,:])
    b=tmp[:,:3*(NV-1)].transpose().dot(Y)
    X=np.zeros(np.shape(SRC))
    for i in range(len(SRC[0,:])):
        X[:3*(NV-1),i]=sl.spsolve(I,sp.csc_matrix(b[:,i]))
        X[3*(NV-1):,i]=SRC[3*(NV-1):,i]
        CreateMesh(np.reshape(X[:,i],(NV,3)),fcs,1)
    print("DT ....", time.time()-CC)
    return X#A-B,B,PsdA 
#####################################

def PICloseForm(X,TInpt,F,NV,NPs,NF,NVF):    
    
    S=np.zeros([3*NV,NPs])
    S[:,0]=TInpt[:,0]
    S[:,NPs-1]=TInpt[:,1]

    #PR=np.roll(P,1,axis=1)
    #PL=np.roll(P,-1,axis=1)
    #grd=X.dot(2*P-PR-PL)
    PR=np.roll(X,1,axis=1)
    PL=np.roll(X,-1,axis=1)
    grd=2*X-PR-PL


    w=1.9
    for itr in range(100):
        for ps in range(1,NPs-1):
            S[:,ps]=(1-w)*S[:,ps]+(w/2)*(grd[:,ps]+S[:,ps-1]+S[:,ps+1])
    
    V_out=np.zeros([NV,3*(NPs-2)])
    
    for i in range(NPs-2):
        V_out[:,3*i:3*i+3]=np.reshape(S[:,i+1],[NV,3])
    
    CreateMesh(V_out,F,NPs-2)

    return

#####################################################################################################
#           Deformation Transfer
#####################################################################################################

class DTToolsPanel(bpy.types.Panel):
    bl_label = "DT Tools Panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "TOOLS"
 
    def draw(self, context):
        self.layout.operator("get.seq",text='Source Seq').seqType="source"
        self.layout.operator("get.seq",text='Target Seq').seqType="target"
        self.layout.prop(context.scene,"FaceFilePath")
        self.layout.operator("get.face",text='Faces').seqType="faces"
        self.layout.operator("dt.tools",text='Preprocessing').seqType="Initialization"
        self.layout.operator("dt.tools",text='DTPI').seqType="DTPI"
        
        
class GetSequence(bpy.types.Operator):
    bl_idname = "get.seq"
    bl_label = "Target Sequence"
    seqType = bpy.props.StringProperty()
 
    def execute(self, context):
        path=bpy.utils.resource_path('USER')
        Selected_Meshes=bpy.context.selected_objects
        obj = bpy.context.active_object
        V=np.zeros([3*len(obj.data.vertices),len(Selected_Meshes)])
        
        for i in range(len(Selected_Meshes)):
            bpy.context.scene.objects.active = Selected_Meshes[-i-1]
            obj = bpy.context.active_object
             
            t=0
            for v in obj.data.vertices:
                co_final= obj.matrix_world*v.co
                V[3*t:3*t+3,i]=np.array([co_final.x,co_final.z,-co_final.y])
                t+=1
        np.savetxt(path+self.seqType+'_vertz.txt',V,delimiter=',')                   
        return{'FINISHED'}
    
class GetFaces(bpy.types.Operator):
    bl_idname = "get.face"
    bl_label = "Target Sequence"
    seqType = bpy.props.StringProperty()
 
    def execute(self, context):
        path=bpy.utils.resource_path('USER')
        FacePath=context.scene.FaceFilePath
        Selected_Meshes=bpy.context.selected_objects
        obj = bpy.context.active_object
        if FacePath=='Default':
            if len(obj.data.polygons)==0:
                F=np.array([[0,1],[1,2], [1,3], [2,4], [3,5], [4,6], [5,7], [6,8],
                             [7,9],[1,10], [10,11], [11,12], [11,13], [12,14], [13,15],
                             [14,16], [15,17], [16,18], [17,19]])
            else:
                F=np.zeros([len(obj.data.polygons),len(obj.data.polygons[0].vertices)],dtype=int)
                t=0
                for f in obj.data.polygons:
                    F[t,:]=f.vertices[:]
                    t+=1
        else:
            F=[]
            file=open(FacePath,'r')
            for line in file:
                words=line.split()
                tmp=[]
                for i in range(len(words)):
                    tmp.append(int(words[i]))
                F.append(tmp)
        np.savetxt(path+'facez.txt',F,delimiter=',')               
        return{'FINISHED'}


def save_sparse_csc(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)

def load_sparse_csc(filename):
    loader = np.load(filename)
    return sp.csc_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])

class DeformationTransferTools(bpy.types.Operator):
    bl_idname = "dt.tools"
    bl_label = "DT Tools"
    seqType = bpy.props.StringProperty()
    
    def execute(self,context):
        path=bpy.utils.resource_path('USER')
        sourceInpt=np.loadtxt(path+'source_vertz.txt',delimiter=',')
        trgtInpt=np.loadtxt(path+'target_vertz.txt',delimiter=',')
        F=np.loadtxt(path+'facez.txt',delimiter=',').astype(int)

        NV,NPs=np.shape(sourceInpt)
        NV=int(NV/3)
        NF,NVF=np.shape(F)

        t=np.size(trgtInpt)/len(trgtInpt)
        Trgt=np.zeros([len(trgtInpt),2])
        if t>=2:
            Trgt[:,0]=trgtInpt[:,0]
            Trgt[:,1]=trgtInpt[:,t-1]
        else:
            Trgt[:,0]=trgtInpt
            Trgt[:,1]=trgtInpt
            NPs+=1
        
        if self.seqType=="Initialization":
            X=ConnectionMatrices(sourceInpt,np.reshape(Trgt[:,0],(NV,3)),F,NV,NF,NVF)
            #save_sparse_csc(path+'/X',X.tocsc())
            np.savetxt(path+'/X.txt',X,delimiter=',')
        else:
            #X=load_sparse_csc(path+'/X.npz')
            #PICloseForm(X, sourceInpt, Trgt, F, NV, NPs, NF, NVF)
            CC=time.time()
            X=np.loadtxt(path+'/X.txt',delimiter=',')
            PICloseForm(X, Trgt, F, NV, NPs, NF, NVF)
            print("PI ....", time.time()-CC)
       
        return {'FINISHED'}


def register():
    bpy.utils.register_class(DTToolsPanel)
    bpy.utils.register_class(GetSequence)
    bpy.utils.register_class(GetFaces)
    bpy.utils.register_class(DeformationTransferTools)
    bpy.types.Scene.FaceFilePath=bpy.props.StringProperty(name="Face Path", description="My Faces", default="Default")
    
   

def unregister():
    bpy.utils.unregister_class(DTToolsPanel)
    bpy.utils.unregister_class(GetSequence)
    bpy.utils.unregister_class(GetFaces)
    bpy.utils.unregister_class(DeformationTransferTools)
    del bpy.types.Scene.FaceFilePath
 
if __name__ == "__main__":  # only for live edit.
    bpy.utils.register_module(__name__) 


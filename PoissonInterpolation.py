bl_info = {
    "name": "Poisson Interpolation",
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
from scipy import sparse as sp
from scipy.sparse import linalg as sl


#########################################################################################
#                   Display Mesh
#########################################################################################

def CreateMesh(V,F,NPs):
    E=np.zeros(np.shape(V))
    F = [[int(i) for i in F]]
    for i in range(NPs-2):
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

def ConnectionMatrices(src,trgt,fcs,NV,NF,NVF):
    Phi=sp.lil_matrix((3*NVF*NF,3*NVF*NF))
    B=sp.lil_matrix((3*NVF*NF,3*NV))
    A=sp.lil_matrix((3*NVF*NF,3*NV))
    D=sp.lil_matrix((3*NF,3*NVF*NF))
    L=[[]]*NV
    
    
    Q=src-np.mean(src,axis=0)
    T=trgt-np.mean(trgt,axis=0)

    Phi[:,:]=FaceTransform(T,Q,NVF)    
    for i in range(NVF):
        B[:,3*i:3*i+3]=(np.array([[1,0,0],[0,1,0],[0,0,1]]*NVF,dtype=float))/NVF
        A[3*i:3*i+3,3*fcs[i]:3*fcs[i]+3]=np.eye(3)             
        

    M=Phi.dot(A-B)+B
    PsdA=(sl.inv(A.transpose().tocsc().dot(A))).dot(A.transpose().tocsc())
    
    return PsdA.dot(M)
#####################################

def PICloseForm(P,TInpt,F): 
    NF=1   
    NVF,=np.shape(F)
    NV,NPs=np.shape(P)
    NV=int(NV/3)
    TrgtInpt=np.zeros([3*NV,2])
    TrgtInpt[:,0]=TInpt[:,0]
    if len(TInpt[0,:])==1:
        TrgtInpt[:,1]=TInpt[:,0]
    else:
        TrgtInpt[:,1]=TInpt[:,len(TInpt[0,:])-1]

    nop,TrNPs=np.shape(TrgtInpt)
    S=np.zeros([3*NV,NPs])
    S[:,0]=TrgtInpt[:,0]
    S[:,NPs-1]=TrgtInpt[:,TrNPs-1]

    P1=np.reshape(P[:,0],[NV,3])
    S1=np.reshape(S[:,0],[NV,3])
    V_out=np.zeros([NV,3*(NPs-2)])

    X=ConnectionMatrices(P1,S1,F,NV,NF,NVF)
   
    PR=np.roll(P,1,axis=1)
    PL=np.roll(P,-1,axis=1)
    grd=X.dot(2*P-PR-PL)


    w=1.9
    for itr in range(100):
        for ps in range(1,NPs-1):
            S[:,ps]=(1-w)*S[:,ps]+(w/2)*(grd[:,ps]+S[:,ps-1]+S[:,ps+1])
    
    for i in range(1,NPs-1):
        V_out[:,3*(i-1):3*(i-1)+3]=np.reshape(S[:,i],[NV,3])

    CreateMesh(V_out,F,NPs)

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
        self.layout.operator("dt.tools",text='DTPI').seqType="PI" 
        

# Operator
class GetSequence(bpy.types.Operator):
    bl_idname = "get.seq"
    bl_label = "Target Sequence"
    seqType = bpy.props.StringProperty()
 
    def execute(self, context):
        path='/home/prashant/Dropbox/poisson_mesh_temporal/Code/Blender/'
        Selected_Meshes=bpy.context.selected_objects
        obj = bpy.context.active_object
        F=np.zeros([len(obj.data.polygons),len(obj.data.polygons[0].vertices)],dtype=int)
        V=np.zeros([3*len(obj.data.vertices),len(Selected_Meshes)])
        t=0
        for f in obj.data.polygons:
                F[t,:]=f.vertices[:]
                t+=1
        
        for i in range(len(Selected_Meshes)):
            bpy.context.scene.objects.active = Selected_Meshes[i]
            obj = bpy.context.active_object
             
            t=0
            for v in obj.data.vertices:
                co_final= obj.matrix_world*v.co
                V[3*t:3*t+3,i]=np.array([co_final.x,co_final.z,-co_final.y])
                t+=1
        np.savetxt(path+self.seqType+'_vertz.txt',V,delimiter=',')
        np.savetxt(path+'facez.txt',F,delimiter=',')                      
        return{'FINISHED'}    
 

class DeformationTransferTools(bpy.types.Operator):
    bl_idname = "dt.tools"
    bl_label = "DT Tools"
    seqType = bpy.props.StringProperty()
    def execute(self,context):
        path='/home/prashant/Dropbox/poisson_mesh_temporal/Code/Blender/'  
        sourceInpt=np.loadtxt(path+'source_vertz.txt',delimiter=',')
        TrgtInpt=np.loadtxt(path+'target_vertz.txt',delimiter=',')
        F=np.loadtxt(path+'facez.txt',delimiter=',').astype(int)
 
        PICloseForm(sourceInpt,TrgtInpt,F)
        return {'FINISHED'}

def register():
    bpy.utils.register_class(DTToolsPanel)
    bpy.utils.register_class(GetSequence)
    bpy.utils.register_class(DeformationTransferTools)
    
   

def unregister():
    bpy.utils.unregister_class(DTToolsPanel)
    bpy.utils.unregister_class(GetSequence)
    bpy.utils.unregister_class(DeformationTransferTools)
 
if __name__ == "__main__":  # only for live edit.
    bpy.utils.register_module(__name__) 



bl_info = {
    "name": "Guided Deformation Transfer",
    "author": "Prashant Domadiya",
    "version": (1, 3),
    "blender": (2, 71, 0),
    "location": "Python Console > Console > Run Script",
    "description": "Transfer Deformation From Sorce Temporal Sequence to Target Temporal Sequence",
    "warning": "",
    "wiki_url": "",
    "tracker_url": "",
    "category": "Development"}

import sys
sys.path.append('/home/prashant/anaconda3/envs/Blender269/lib/python3.4/site-packages')

import bpy
import numpy as np

import os
from os.path import join

import os
from scipy import sparse as sp
from sksparse import cholmod as chmd

from functools import reduce,partial
from multiprocessing import Pool
import time
import itertools as itr
import shutil


##############################################################################################
#                      Functionas
##############################################################################################

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #                   Display Mesh
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


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

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #               Face Transformation
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


 
def SkelTransform(MM,NPs,NVert,FacesVertex):
    NVF=int(FacesVertex[len(FacesVertex)-2])
    FaceNum=int(FacesVertex[len(FacesVertex)-1])
    SrcIni=np.reshape(FacesVertex[0:3*NVF],(NVF,3))
    SrcIni=SrcIni-np.mean(SrcIni,axis=0)
    SrcIni=np.array((SrcIni.tolist())*(NPs-1))
    
    Src=np.reshape(FacesVertex[3*NVF:len(FacesVertex)-2],(NVF*(NPs-1),3))
    for i in range(NPs-1):
        Src[NVF*i:NVF*i+NVF,:]=Src[NVF*i:NVF*i+NVF,:]-np.mean(Src[NVF*i:NVF*i+NVF,:],axis=0)

    
    Scl=np.linalg.norm(Src,axis=1)/np.linalg.norm(SrcIni,axis=1)
    W=np.cross(SrcIni,Src)
    WNorm=np.linalg.norm(W,axis=1)
    Dot_prdct=np.sum(SrcIni*Src,axis=1)/(np.linalg.norm(SrcIni,axis=1)*np.linalg.norm(Src,axis=1))
    
    R=np.zeros([3*NVF,3*NVF])
    V22=np.zeros([3*NVF,(NPs-1)])
    for l in range(NPs-1):
        for i in range(NVF):
            if WNorm[NVF*l+i]==0:
                R[3*i:3*i+3,3*i:3*i+3]=Scl[i]*np.eye(3)
            else:
                w=W[NVF*l+i,:]/WNorm[NVF*l+i]
                if Dot_prdct[NVF*l+i]>1.0:
                    C=1.0
                    S=0.0
                elif Dot_prdct[NVF*l+i]<-1.0:
                    C=-1.0
                    S=0.0
                else:
                    C=Dot_prdct[NVF*l+i]
                    S=np.sin(np.arccos(Dot_prdct[NVF*l+i]))     
                T=1-C
                R[3*i:3*i+3,3*i:3*i+3]=Scl[NVF*l+i]*np.array([[T*(w[0]**2)+C,T*w[0]*w[1]-S*w[2], T*w[0]*w[2]+S*w[1]],[T*w[0]*w[1]+S*w[2],T*(w[1]**2)+C,T*w[1]*w[2]-S*w[0]],[T*w[0]*w[2]-S*w[1],T*w[1]*w[2]+S*w[0],T*(w[2]**2)+C]])
        Lol=MM[3*FaceNum:3*FaceNum+3*NVF]
        V22[:,l]=np.dot(R,Lol)
    return V22

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #                   Pose Transformation
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



def PoseTransform(MM,NPs,FacesVertex):
    
    NVF=int(FacesVertex[len(FacesVertex)-2])
    FaceNum=int(FacesVertex[len(FacesVertex)-1])
   
    SrcIni=np.reshape(FacesVertex[0:3*NVF],(NVF,3))
    SrcIni=SrcIni-np.mean(SrcIni,axis=0)

    SX=np.zeros([3*NVF,3])
    SN=np.cross(SrcIni[0],SrcIni[1])/np.linalg.norm(np.cross(SrcIni[0],SrcIni[1]))
    SrcUnit=SrcIni/np.reshape(np.linalg.norm(SrcIni,axis=1),(NVF,1))
    
    Src=np.reshape(FacesVertex[3*NVF:len(FacesVertex)-2],(NVF*(NPs-1),3))
    for i in range(NPs-1):
        Src[NVF*i:NVF*i+NVF,:]=Src[NVF*i:NVF*i+NVF,:]-np.mean(Src[NVF*i:NVF*i+NVF,:],axis=0)
    SrcUnit1=Src/np.reshape(np.linalg.norm(Src,axis=1),(NVF*(NPs-1),1))

    Theta=np.zeros([3*NVF,3*NVF])
    V22=np.zeros([3*NVF,(NPs-1)])
    Nrml=np.zeros([3,NPs])
    Lol=MM[3*FaceNum:3*FaceNum+3*NVF]
  
    Loltemp=np.reshape(Lol,(NVF,3))
    Nrml[:,0]=np.mean(np.cross(Loltemp,np.roll(Loltemp,-1,axis=0)).T/np.linalg.norm(np.cross(Loltemp,np.roll(Loltemp,-1,axis=0)),axis=1),axis=1)
    
    #Nrml[:,0]= np.cross(Lol[0:3],Lol[3:6])/np.linalg.norm(np.cross(Lol[0:3],Lol[3:6]))

    for l in range(NPs-1):
        SX1=np.zeros([3,3])
        SN1=np.cross(Src[l*NVF],Src[l*NVF+1])/np.linalg.norm(np.cross(Src[l*NVF],Src[l*NVF+1]))
                                             
        for i in range(NVF):
            if l==0:
                SX[3*i,:]=SrcUnit[i]
                SX[3*i+1,:]=np.cross(SN,SX[3*i])/np.linalg.norm(np.cross(SN,SX[3*i]))
                SX[3*i+2,:]=np.cross(SX[3*i],SX[3*i+1])/np.linalg.norm(np.cross(SX[3*i],SX[3*i+1]))
            SX1[:,0]=SrcUnit1[l*NVF+i]
            SX1[:,1]=np.cross(SN1,SX1[:,0])/np.linalg.norm(np.cross(SN1,SX1[:,0]))
            SX1[:,2]=np.cross(SX1[:,0],SX1[:,1])/np.linalg.norm(np.cross(SX1[:,0],SX1[:,1]))
            
            Scl=np.linalg.norm(Src[l*NVF+i])/np.linalg.norm(SrcIni[i])
            Theta[3*i:3*i+3,3*i:3*i+3]=Scl*np.dot(SX1,SX[3*i:3*i+3,:])
        
        V22[:,l]=np.dot(Theta,Lol)
        V22temp=np.reshape(V22[:,l],(NVF,3))
        Nrml[:,l+1]=np.mean(np.cross(V22temp,np.roll(V22temp,-1,axis=0)).T/np.linalg.norm(np.cross(V22temp,np.roll(V22temp,-1,axis=0)),axis=1),axis=1)

        #Nrml[:,l+1]=np.cross(V22[0:3,l],V22[3:6,l])/np.linalg.norm(np.cross(V22[0:3,l],V22[3:6,l]))
    return V22,Nrml

def ConnectionMatrices(trgt,fcs,CrT,NV,NVert,NPs):
    
    B=sp.lil_matrix((3*NVert,3*NV),dtype=float)
    A=sp.lil_matrix((3*NVert,3*NV),dtype=float)
    V2=np.zeros([3*NVert,NPs])
    tt=0
    
    for t in CrT:
        NVF=len(fcs[t])
        FF=np.reshape(3*np.array([fcs[t]]*3)+np.array([[0],[1],[2]]),3*NVF,order='F')
        B[tt:tt+3*NVF,FF]=(np.array([[1,0,0]*NVF,[0,1,0]*NVF,[0,0,1]*NVF]*NVF,dtype=float))/NVF
        A[tt:tt+3*NVF,FF]=np.eye(3*NVF)
        tt+=3*NVF
    
    V2[:,0]=(A-B).dot(trgt)
    return A,B,V2

def GetTranslation(D,I,NV,NVert,CrT,TF,Inpt):
    b=Inpt[0:3*(NV-1)]
    Nrml=Inpt[3*(NV-1):len(Inpt)-3]
    C=Inpt[len(Inpt)-3:]
    M=sp.lil_matrix((NVert,3*NVert))
    t=0
    tt=0
    for f in CrT:
        if len(TF[f])>3:
            for j in range(len(TF[f])):
                M[t,3*t:3*t+3]=Nrml[3*tt:3*tt+3]
                t+=1
        else:
            t+=3
        tt+=1
    E=M.dot(D)
    E_hat=(E[:,:3*(NV-1)].transpose()).dot(E[:,:3*(NV-1)])
    Y1=-E[:,3*NV-3:].dot(C)
    b1=E[:,:(3*NV-3)].transpose().dot(Y1)
    factor=chmd.cholesky(0.05*I+E_hat)
    X=factor(sp.csc_matrix(np.reshape(0.05*b+b1,(len(b),1))))
    X=np.append(np.ravel(X.todense()),C)
    return X
    

def PICloseForm(SInpt,TInpt,PoseInpt,SCen,TF,CrT):
    
    NV=len(TInpt)
    NPs=len(SInpt.T)
    NV=int(NV/3)
    FLen=[len(a) for a in TF]
    NVert=sum([len(TF[i]) for i in CrT])                                      #####
    NF=len([len(TF[i]) for i in CrT])                                         #####

    if len([1 for i in FLen if i > 3])!=0:
        FaceType='quad'
    else:
        FaceType='Tris'
    
    strtime=time.time()
    A,B,V2=ConnectionMatrices(TInpt,TF,CrT,NV,NVert,NPs)
    
    
    print("Connection matrices ...", time.time()-strtime)
    
    #-----------------------Pose Transform------------------------------------
    # Assumtion: src and trgt are of the form m x 3
    #-------------------------------------------------------------------------
    strtime=time.time()
    p=Pool()
    if len(TF[0])==2:
        func=partial(SkelTransform,V2[:,0],NPs,NVert)
        PrllOut=p.map(func,PoseInpt)

        t=0
        count=0
        for i in CrT:
            V2[3*t:3*t+3*FLen[i],1:]= PrllOut[count]
            count+=1
            t+=FLen[i]    
    else:
        func=partial(PoseTransform,V2[:,0],NPs)
        PrllOut=p.map(func,PoseInpt)

        Nrml=np.zeros([3*len(CrT),NPs])
        t=0
        count=0
        for i in CrT:
            V2[3*t:3*t+3*FLen[i],1:]= PrllOut[count][0]
            Nrml[3*count:3*count+3]=PrllOut[count][1]
            count+=1
            t+=FLen[i]    
    p.close()

    D=(A-B)
    I=(D[:,:3*(NV-1)].transpose()).dot(D[:,:3*(NV-1)])
    C= SInpt[3*(NV-1):,:]
    Y=V2-D[:,3*NV-3:].dot(C)
    b=D[:,:(3*NV-3)].transpose().dot(Y)
    
    SRC=np.zeros([3*NV,NPs])
    if FaceType=="quad":
        print("processing Quad ...")

        p=Pool()
        func=partial(GetTranslation,D,I,NV,NVert,CrT,TF)
        PrllInpt=(np.concatenate((b,Nrml,C),axis=0)).T
        
        PrllOut=p.map(func,PrllInpt)

        SRC=np.zeros([3*NV,NPs])
        for i in range(NPs):
            temp=np.reshape(PrllOut[i],[NV,3])
            SRC[:,i]=np.ravel(temp+SCen[3*i:3*i+3]-np.mean(temp,axis=0))
        p.close()
    else:
        factor=chmd.cholesky(I.tocsc())
        X=factor(b)
        X=np.append(X,C,axis=0)
        for i in range(NPs):
            temp=np.reshape(X[:,i],[NV,3])
            SRC[:,i]=np.ravel(temp+SCen[3*i:3*i+3]-np.mean(temp,axis=0))
        
    print("Computation time ....", time.time()-strtime)
    
    V_out=np.zeros([NV,3*NPs])    
    for i in range(NPs):
        V_out[:,3*i:3*i+3]=np.reshape(SRC[:,i],[NV,3])
        
    CreateMesh(V_out,TF,int(NPs))        
            
    return SRC


############################################################
def PI(SRC,TInpt,TFori):
    NV,NPs=np.shape(SRC)
    NV=int(NV/3)

    P=np.zeros([3*NV,NPs])
    TrNPs=np.size(TInpt)/(3*NV)
    if TrNPs==1:
        P[:,0]=1*TInpt
    else:
        P[:,0],P[:,NPs-1],TrNPs=TInpt[:,0],TInpt[:,TrNPs-1],2
    
    if TrNPs==1:
        P=np.append(P,np.reshape(P[:,0],(3*NV,1)),axis=1)
        SRC=np.append(SRC,np.reshape(SRC[:,0],(3*NV,1)),axis=1)
        NPs+=1
        TrNPs+=1
    
    PR=np.roll(SRC,1,axis=1)
    PL=np.roll(SRC,-1,axis=1)
    grd=2*SRC-PR-PL
    
    w=1.9
    for itr in range(100):
        for ps in range(1,NPs-1):
            P[:,ps]=(1-w)*P[:,ps]+(w/2)*(grd[:,ps]+P[:,ps-1]+P[:,ps+1])
      
    V_out=np.zeros([NV,3*int(NPs-TrNPs)])    
    for i in range(int(NPs-TrNPs)):
        V_out[:,3*i:3*i+3]=np.reshape(P[:,i+1],[NV,3])
    
    CreateMesh(V_out,TFori,int(NPs-TrNPs))
    return

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
    CrS=[[]]*len(TF)
    fl = open(fileName, 'r')
    for line in fl:
        words = line.split()
        l=len(words)
        CrT.append(int(words[1]))
        CrS[int(words[1])]=CrS[int(words[1])]+[int(words[0])]
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
#####################################################################################################
#           Deformation Transfer Tool
#####################################################################################################

class DTToolsPanel(bpy.types.Panel):
    bl_label = "DT Tools Panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "TOOLS"
 
    def draw(self, context):
        self.layout.operator("get.seq",text='Source Seq').seqType="source"
        self.layout.operator("get.seq",text='Target Seq').seqType="target"
        self.layout.prop(context.scene,"CorrPath")
        self.layout.operator("dt.tools",text='DT').seqType="DT"
        self.layout.operator("dt.tools",text='PI').seqType="PI"
        


class GetSequence(bpy.types.Operator):
    bl_idname = "get.seq"
    bl_label = "Target Sequence"
    seqType = bpy.props.StringProperty()
 
    def execute(self, context):
        path=bpy.utils.resource_path('USER') # '/home/student/Dropbox/My_PC/poisson_mesh_temporal/Code/Blender/'#
        
        Selected_Meshes=bpy.context.selected_objects
        obj = bpy.context.active_object
        V=np.zeros([3*len(obj.data.vertices),len(Selected_Meshes)])
        
        NPs=len(Selected_Meshes)
        for i in range(len(Selected_Meshes)):
            bpy.context.scene.objects.active = Selected_Meshes[-i-1]
            obj = bpy.context.active_object
             
            t=0
            for v in obj.data.vertices:
                co_final= obj.matrix_world*v.co
                V[3*t:3*t+3,i]=np.array([co_final.x,co_final.z,-co_final.y])
                t+=1
        F=[]
        if len(obj.data.polygons)!=0:
            for f in obj.data.polygons:
                F.append(list(f.vertices))
        else:
            for e in obj.data.edges:
                F.append(list(e.vertices))
        np.savetxt(path+'/'+self.seqType+'_vertz.txt',V,delimiter=',')
        WriteAsTxt(path+'/'+self.seqType+'_facz.txt',F)
        return{'FINISHED'}



class DeformationTransferTools(bpy.types.Operator):
    bl_idname = "dt.tools"
    bl_label = "DT Tools"
    seqType = bpy.props.StringProperty()
    def execute(self,context):
        path=bpy.utils.resource_path('USER') #'/home/student/Dropbox/My_PC/poisson_mesh_temporal/Code/Blender/'#
        
        SourceInpt=np.loadtxt(path+'/'+'source_vertz.txt',delimiter=',')
        TrgtInpt=np.loadtxt(path+'/'+'target_vertz.txt',delimiter=',')
         
        SF=ReadTxt(path+'/'+'source_facz.txt')
        TF=ReadTxt(path+'/'+'target_facz.txt')
        if len(SourceInpt)==len(TrgtInpt):
            SF=TF

        if context.scene.CorrPath!='Default':
            CrT,CrS=ReadCorrespondences(context.scene.CorrPath,TF)
        else:
            CrT=range(len(TF))
            CrS=[[i] for i in range(len(SF))]
            SF=TF
        

        NPs=len(SourceInpt[0,:])
        TrNPs=int(np.size(TrgtInpt)/len(TrgtInpt))
        if self.seqType=='DT':
            O=np.zeros(3*(NPs))
            for i in range(NPs):
                O[3*i:3*i+3]=np.mean(np.reshape(SourceInpt[:,i],(int(len(SourceInpt)/3),3)),axis=0)
            PoseFace=GetFaceVrtz(SourceInpt,SF,TF,CrS,CrT)
            
            if TrNPs==1:
                
                SRC=PICloseForm(SourceInpt,TrgtInpt,PoseFace,O,TF,CrT)
                np.savetxt(path+'/'+'Target_Deformed.txt',SRC,delimiter=',')
                
            else:
                
                SRC=PICloseForm(SourceInpt,TrgtInpt[:,0],PoseFace,O,TF,CrT)
                np.savetxt(path+'/'+'Target_Deformed.txt',SRC,delimiter=',')
                
        else:
            TrgtDfrm=np.loadtxt(path+'/'+'Target_Deformed.txt',delimiter=',')
            PI(TrgtDfrm,TrgtInpt,TF)
        
        return {'FINISHED'}


   

def register():
    bpy.utils.register_class(DTToolsPanel)
    bpy.utils.register_class(GetSequence)
    bpy.types.Scene.CorrPath=bpy.props.StringProperty(name="Correspondence", description="Face Correspondences", default="Default")
    bpy.utils.register_class(DeformationTransferTools)
    
    
    
def unregister():
    bpy.utils.unregister_class(DTToolsPanel)
    bpy.utils.unregister_class(GetSequence)
    bpy.utils.unregister_class(DeformationTransferTools)
    del bpy.types.Scene.CorrPath
    
 
if __name__ == "__main__":  # only for live edit.
    bpy.utils.register_module(__name__) 



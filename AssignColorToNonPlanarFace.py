
bl_info = {
    "name": "Assign color to faces",
    "author": "Prashant Domadiya",
    "version": (1, 1),
    "blender": (2, 69, 0),
    "location": "Python Console > Console > Run Script",
    "description": "Coloring faces",
    "warning": "",
    "wiki_url": "",
    "tracker_url": "",
    "category": "Development"}

import bpy
import bmesh
import numpy as np


##############################################################################################
#                      Functionas
##############################################################################################

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #                   Display Mesh
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


################################################################
#           Deformation Transfer Tool
#####################################################################################################

class DTToolsPanel(bpy.types.Panel):
    bl_label = "DT Tools Panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "TOOLS"
 
    def draw(self, context):
        self.layout.operator("get.seq",text='Target Seq')
        
class GetSequence(bpy.types.Operator):
    bl_idname = "get.seq"
    bl_label = "Target Sequence"
    seqType = bpy.props.StringProperty()
 
    def execute(self, context):
        obj = bpy.context.active_object
        V=np.zeros([len(obj.data.vertices),3])

        t=0
        for v in obj.data.vertices:
            co_final= obj.matrix_world*v.co
            V[t]=np.array([co_final.x,co_final.y,co_final.z])
            t+=1
        
        mat_green=bpy.data.materials.new('green')
        mat_green.diffuse_color=[0,0.8,0]
        mat_blue=bpy.data.materials.new('blue')
        mat_blue.diffuse_color=[0,0,0.8]
        mat_red=bpy.data.materials.new('red')
        mat_red.diffuse_color=[0.8,0,0]
        mesh=bpy.context.object.data
        print(mesh)
        mesh.materials.append(mat_green)
        mesh.materials.append(mat_blue)
        mesh.materials.append(mat_red)
        count=0
        bm=bmesh.new()
        bm.from_mesh(mesh)
        for f in bm.faces:
            F=[]
            for v in f.verts:
                    F.append(v.index)
            if len(F)>3:  
                vert=V[F,:]-np.mean(V[F,:],axis=0)
                N=np.cross(vert[0,:],vert[1,:])
                N=N/np.linalg.norm(N)
                NVert=vert.T/np.linalg.norm(vert,axis=1)
                Err=np.mean(abs(np.dot(NVert[:,2:].T,N)))
                
                if Err>0.05 and Err<=0.1:
                    f.material_index=1
                elif Err>0.1:
                    f.material_index=2
            
        bm.to_mesh(mesh)
        return{'FINISHED'}


def register():
    bpy.utils.register_class(DTToolsPanel)
    bpy.utils.register_class(GetSequence)
    
    
def unregister():
    bpy.utils.unregister_class(DTToolsPanel)
    bpy.utils.unregister_class(GetSequence)

 
if __name__ == "__main__":  # only for live edit.
    bpy.utils.register_module(__name__) 



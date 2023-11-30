import pypangolin as pangolin
import OpenGL.GL as gl
import time 
import threading 
import copy
import numpy as np


class Viewer:
    def __init__(self, w, h):
        self.travelets = {}
        self.trajs = []
        self.is_running = True 
        self.w = w 
        self.h = h 
        print(f"w={w}, h={h}")
        
        self.img = None 
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.run)
        self.thread.start()
        self.current_frame = 0


    def run(self):
        pangolin.CreateWindowAndBind('Main', int(self.w * 1.0), 2 * self.h)
        gl.glEnable(gl.GL_DEPTH_TEST)

        scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(self.w, self.h, 400, 400, self.w / 2, self.h / 2, 0.1, 1000),
                        pangolin.ModelViewLookAt(0, -5., -0.1, 0, 0, 0, pangolin.AxisNegY))

        handler = pangolin.Handler3D(scam)

        # Create Interactive View in window
        ui_width = 180
        dcam = pangolin.CreateDisplay()
        dcam.SetBounds(
                pangolin.Attach(0),
                pangolin.Attach(1),
                pangolin.Attach.Pix(ui_width),
                pangolin.Attach(1),
                -self.w * 1.0 / self.h,
            )

        dcam.SetHandler(handler)

        d_video = pangolin.Display("imgVideo").SetAspect(self.w * 1.0 / self.h)
        texVideo = pangolin.GlTexture(self.w, self.h, gl.GL_RGB, False, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        pangolin.CreateDisplay(
                ).SetBounds(
                    pangolin.Attach(0), 
                    pangolin.Attach(0.45), 
                    pangolin.Attach.Pix(10),
                    pangolin.Attach(1.0)
                ).SetLayout(pangolin.Layout.Equal).AddDisplay(d_video)
        
        '''
        pango.CreatePanel("ui").SetBounds(
        pango.Attach(0), pango.Attach(1), pango.Attach(0), pango.Attach.Pix(ui_width)
        )
        var_ui = pango.Var("ui")
        var_ui.a_Button = False
        var_ui.a_double = (0.0, pango.VarMeta(0, 5))
        var_ui.an_int = (2, pango.VarMeta(0, 5))
        var_ui.a_double_log = (3.0, pango.VarMeta(1, 1e4, logscale=True))
        var_ui.a_checkbox = (False, pango.VarMeta(toggle=True))
        var_ui.an_int_no_input = 2
        var_ui.a_str = "sss"
        '''
        pangolin.CreatePanel("menu").SetBounds(pangolin.Attach(0.0), pangolin.Attach(1.0), pangolin.Attach(0.0), pangolin.Attach.Pix(150))
        menu_var = pangolin.Var("menu")
        menu_var.followCamera = (False, pangolin.VarMeta(toggle=True))
        menu_var.showVideo = (True, pangolin.VarMeta(toggle=True))
        menu_var.allObject = (False, pangolin.VarMeta(toggle=True))

        while self.is_running:
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            gl.glClearColor(1.0, 1.0, 1.0, 1.0)
            dcam.Activate(scam)

            if len(self.trajs) == 0:
                time.sleep(0.5)
                continue 
            
            with self.lock:
                pose = self.trajs[-1]
                if menu_var.followCamera:
                    pose = np.vstack([np.hstack([pose['rotation'], pose['position'].reshape(-1, 1)]), np.asarray([0, 0, 0, 1])])
                    pose = pangolin.OpenGlMatrix(pose)
                    scam.Follow(pose, follow=True)
                else:
                    pose = np.vstack([np.hstack([np.eye(3), pose['position'].reshape(-1, 1)]), np.asarray([0, 0, 0, 1])])
                    pose = pangolin.OpenGlMatrix(pose)
                    #scam.Follow(pose)

                for tid in self.travelets:
                    frame_count = self.travelets[tid][-1][3]

                    if frame_count < (self.current_frame - 5) and not menu_var.allObject:
                        continue 

                    locs = [it[0] for it in self.travelets[tid]]
                    color = self.travelets[tid][0][2]
                    gl.glLineWidth(2)
                    gl.glColor3f(color[0], color[1], color[2])
                    gl.glBegin(gl.GL_LINES)
                    for i in range(1, len(locs)):
                        p1 = locs[i-1]
                        p2 = locs[i]

                        gl.glVertex3f(p1[0], p1[1], p1[2])
                        gl.glVertex3f(p2[0], p2[1], p2[2])
                    gl.glEnd()

                    # draw box 
                    box3d = self.travelets[tid][-1][1]
                    
                    lineorder = np.array(
                    [
                        [1, 2, 6, 5],  # front face
                        [2, 3, 7, 6],  # left face
                        [3, 4, 8, 7],  # back face
                        [4, 1, 5, 8],
                        [1, 6, 5, 2]
                    ], dtype=np.int32) - 1  # right

                    planes = [[1, 2], [2, 3], [3, 4], [4, 1], [1, 5], [5, 6], [6, 2], [6, 7], [7, 3], [7, 8], [8, 4], [8, 5], [1, 6], [2, 5]]

                    gl.glLineWidth(2)
                    #gl.glColor3f(0.0, 1.0, 0.0)
                    gl.glColor3f(color[0], color[1], color[2])
                    gl.glBegin(gl.GL_LINES)
                
                    for it in planes:
                        i1, i2 = it
                        p1 = box3d[i1-1]
                        p2 = box3d[i2-1]
                        gl.glVertex3f(p1[0], p1[1], p1[2])
                        gl.glVertex3f(p2[0], p2[1], p2[2])
                
                    gl.glEnd()

                # draw car trajectory
                gl.glLineWidth(3)
                gl.glColor3f(1.0, 0.0, 0.0)
                gl.glBegin(gl.GL_LINES)
                for i in range(1, len(self.trajs)):
                    p1 = self.trajs[i-1]['position']
                    p2 = self.trajs[i]['position']

                    gl.glVertex3f(p1[0], p1[1], p1[2])
                    gl.glVertex3f(p2[0], p2[1], p2[2])
                gl.glEnd()

                if len(self.trajs) > 0:
                    # draw camera
                    pose = self.trajs[-1]
                    pose = np.vstack([np.hstack([pose['rotation'], pose['position'].reshape(-1, 1)]), np.asarray([0, 0, 0, 1])])

                    gl.glPushMatrix()
                    #gl.glMultMatrixf(pose)
                    gl.glMultTransposeMatrixf(pose)

                    gl.glLineWidth(3)
                    gl.glColor3f(0.0, 1.0, 0.0)
                    gl.glBegin(gl.GL_LINES)

                    w, h, z = 0.5, 0.75, 0.8
                    h = w * h * 1.5
                    z = w * h * 1.5
                    w, h, z = np.array([w, h, z]) * 3.0

                    gl.glVertex3f(0,0,0)
                    gl.glVertex3f(w,h,z)

                    gl.glVertex3f(0,0,0)
                    gl.glVertex3f(w,-h,z)

                    gl.glVertex3f(0,0,0)
                    gl.glVertex3f(-w,-h,z)

                    gl.glVertex3f(0,0,0)
                    gl.glVertex3f(-w,h,z)

                    gl.glVertex3f(w,h,z)
                    gl.glVertex3f(w,-h,z)

                    gl.glVertex3f(-w,h,z)
                    gl.glVertex3f(-w,-h,z)

                    gl.glVertex3f(-w,h,z)
                    gl.glVertex3f(w,h,z)

                    gl.glVertex3f(-w,-h,z)
                    gl.glVertex3f(w,-h,z)
                
                    gl.glEnd()
                    gl.glPopMatrix()


                # image
                if self.img is not None and menu_var.showVideo:
                    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT,1)
                    texVideo.Upload(self.img, gl.GL_BGR, gl.GL_UNSIGNED_BYTE)
                    d_video.Activate()
                    gl.glColor4f(1.0, 1.0, 1.0, 1.0)
                    texVideo.RenderToViewportFlipY()
                

            time.sleep(0.1)
            pangolin.FinishFrame()

    def update(self, travelets, pose, img, current_frame):

        with self.lock:
            self.travelets = copy.deepcopy(travelets)
            self.trajs.append(pose)
            self.img = img.copy()
            self.current_frame = current_frame


    def shutdown(self):
        self.is_running = False 
        self.thread.join()
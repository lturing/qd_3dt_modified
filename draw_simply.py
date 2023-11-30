
# https://github.com/uoip/pangolin
# https://github.com/stevenlovegrove/Pangolin

import OpenGL.GL as gl
import pypangolin as pangolin

import numpy as np



def main():
    pangolin.CreateWindowAndBind('Main', 640, 480)
    gl.glEnable(gl.GL_DEPTH_TEST)

    # Define Projection and initial ModelView matrix
    scam = pangolin.OpenGlRenderState(
        pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 200),
        pangolin.ModelViewLookAt(-2, 2, -2, 0, 0, 0, pangolin.AxisDirection.AxisY))
    handler = pangolin.Handler3D(scam)

    # Create Interactive View in window
    ui_width = 180
    dcam = pangolin.CreateDisplay()
    dcam.SetBounds(
            pangolin.Attach(0),
            pangolin.Attach(1),
            pangolin.Attach.Pix(ui_width),
            pangolin.Attach(1),
            -640.0 / 480.0,
        )

    dcam.SetHandler(handler)

    trajectory = [[0, -6, 6]]
    for i in range(300):
        trajectory.append(trajectory[-1] + np.random.random(3)-0.5)
    trajectory = np.array(trajectory)
    
    boxes = []
    poses = []
    while not pangolin.ShouldQuit():
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        dcam.Activate(scam)
        
        # Render OpenGL Cube
        pangolin.glDrawColouredCube(0.1)
        
        # Draw Point Cloud
        points = np.random.random((10000, 3)) * 3 - 4
        gl.glPointSize(1)
        gl.glBegin(gl.GL_POINTS);
        gl.glColor3f(1.0, 0.0, 0.0)
        for p in points:
            gl.glVertex3f(p[0], p[1], p[2])
            
        gl.glEnd()

        # Draw Point Cloud
        points = np.random.random((10000, 3))
        points = points * 3 + 1
        gl.glPointSize(1)
        gl.glBegin(gl.GL_POINTS);
        gl.glColor3f(0.0, 0.0, 1.0)
        for p in points:
            gl.glVertex3f(p[0], p[1], p[2])
        gl.glEnd()

        # Draw lines
        gl.glLineWidth(1)
        gl.glColor3f(0.0, 0.0, 0.0)
        gl.glBegin(gl.GL_LINES)
        for i in range(1, len(trajectory)):
            p1 = trajectory[i-1]
            p2 = trajectory[i]
            gl.glVertex3f(p1[0], p1[1], p1[2])
            gl.glVertex3f(p2[0], p2[1], p2[2])
        gl.glEnd()
        
        
        # Draw camera
        pose = np.identity(4)
        #pose[:3, 3] = np.random.randn(3)

        gl.glPushMatrix()
        gl.glMultTransposeMatrixf(pose)

        gl.glLineWidth(1)
        gl.glColor3f(0.0, 1.0, 0.0)
        gl.glBegin(gl.GL_LINES)
        
        w, h, z = 0.5, 0.75, 0.8
        h = w * h
        z = w * h

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
        
        
        # Draw boxes
        num_boxes = 10
        for i in range(num_boxes):
            if len(boxes) == num_boxes:
                w, h, z = boxes[i]
                pose = poses[i]
            else:
                w, h, z = np.random.random(3)
                boxes.append([w, h, z])
                pose = np.identity(4)
                pose[:3, 3] = np.random.randn(3) + np.array([5, -3, 2])
                poses.append(pose)
                
            points = [[-w, h, z], [w, h, z], [w, -h, z], [-w, -h, z],
                      [-w, h, -z], [w, h, -z], [w, -h, -z], [-w, -h, -z]
                     ]
            

            planes = [[1, 2], [2, 3], [3, 4], [4, 1], [1, 5], [5, 6], [6, 2], [6, 7], [7, 3], [7, 8], [8, 4], [8, 5]]
            
            gl.glPushMatrix()
            #gl.glMultMatrixf(pose)
            gl.glMultTransposeMatrixf(pose)

            gl.glLineWidth(1)
            gl.glColor3f(0.0, 1.0, 0.0)
            gl.glBegin(gl.GL_LINES)
            
            for it in planes:
                i1, i2 = it
                p1 = points[i1-1]
                p2 = points[i2-1]
                gl.glVertex3f(p1[0], p1[1], p1[2])
                gl.glVertex3f(p2[0], p2[1], p2[2])
            
            gl.glEnd()
            gl.glPopMatrix()

        pangolin.FinishFrame()
    


if __name__ == '__main__':
    main()

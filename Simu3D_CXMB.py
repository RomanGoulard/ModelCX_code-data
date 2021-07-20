#!/usr/bin/env python
import os
import math
import cv2
from pyglet.gl import *
from OpenGL.GL import *
import numpy as np
import pyautogui
import pyglet
from PIL import Image
import time
import random
import shutil

RNG = np.random.RandomState(2018)

verticies = ((200, -200, -200),
             (200, 200, -200),
             (-200, 200, -200),
             (-200, -200, -200),
             (200, -200, 200),
             (200, 200, 200),
             (-200, -200, 200),
             (-200, 200, 200))

surfaces = ((0, 1, 2, 3),
            (3, 2, 7, 6),
            (6, 7, 5, 4),
            (4, 5, 1, 0),
            (1, 5, 7, 2),
            (4, 0, 3, 6))

ground_surface = ((200, 0, 200),
                  (200, 0, -200),
                  (-200, 0, -200),
                  (-200, 0, 200),
                  (200, -1, 200),
                  (200, -1, -200),
                  (-200, -1, -200),
                  (-200, -1, 200))

sky_color = (1.0, 1.0, 1.0)
ground_color = (1.0, 1.0, 1.0)


def read_texture(filename):
    size = (512, 512)

    img = Image.open(filename)

    lim = 90# * np.sqrt(2)

    img_RotCoord = [np.linspace(-lim, lim, img.size[0]),
                    np.linspace(-lim, lim, img.size[1])]

    Xdisp = np.linspace(0, 180, img.size[0])
    Ydisp = np.linspace(0, 360, img.size[1]) - 180
    disp_RotCoord = [Xdisp,
                     Ydisp]
    # disp_RotCoord[0][disp_RotCoord[0] > 180] -= 360
    # print(disp_RotCoord)

    img_mat = np.array(img.getdata(), np.uint8).reshape(img.size[0], img.size[1], 3)
    DispMap = np.zeros(img_mat.shape)

    for ix in range(img_mat.shape[0]):
        for iy in range(img_mat.shape[1]):
            CoordInput_theta = np.degrees(np.arctan2(img_RotCoord[0][ix], img_RotCoord[1][iy]))
            CoordInput_rho = np.sqrt(img_RotCoord[0][ix]**2 + img_RotCoord[1][iy]**2)# * np.tan(np.radians(CoordInput_theta))

            Disp_x = int(np.argmin(abs(disp_RotCoord[0][:] - CoordInput_rho)))
            Disp_y = int(np.argmin(abs(disp_RotCoord[1][:] - CoordInput_theta)))

            DispMap[Disp_x, Disp_y, :] = img_mat[ix, iy, :]

    img2 = cv2.resize(DispMap, size, interpolation=cv2.INTER_AREA)
    # img2filt = np.array(img.getdata(), np.float).reshape(img.size[0], img.size[1], 3)

    # img2 = filtImg(img2, 'lowpass', (2, 1.0))
    # img2 = filtImg(img2, 'highpass', (20, 1.0))

    img = Image.fromarray(np.uint8(img2))

    # print(img)

    img_data = np.array(list(img.getdata()), np.uint8)

    # print(np.unique(img_data))

    textID = glGenTextures(1)
    return textID, img_data, img


def Cube(color):
    glBegin(GL_QUADS)
    for surface in surfaces:
        x = 0
        for vertex in surface:
            x += 1
            glColor3fv(color)
            glVertex3fv(verticies[vertex])
    glEnd()


def Cube2(position, width, height, Color=(0, 0, 0)):
    X1 = -(position[1] - width/2)
    X2 = -(position[1] + width/2)
    Y1 = position[0] - width/2
    Y2 = position[0] + width/2
    Z1 = 0
    Z2 = height

    verticies_cube = ((X1, Z1, Y1),
                      (X1, Z1, Y2),
                      (X2, Z1, Y2),
                      (X2, Z1, Y1),
                      (X1, Z2, Y1),
                      (X1, Z2, Y2),
                      (X2, Z2, Y2),
                      (X2, Z2, Y1))

    surfaces_cube = ((0, 1, 2, 3),
                     (0, 1, 5, 4),
                     (1, 2, 6, 5),
                     (2, 3, 7, 6),
                     (3, 0, 4, 7),
                     (4, 5, 6, 7))

    glBegin(GL_QUADS)
    for surface in surfaces_cube:
        x = 0
        for vertex in surface:
            x += 1
            glColor3fv(Color)
            glVertex3fv(verticies_cube[vertex])
    glEnd()


def Surface(color):
    glBegin(GL_QUADS)
    for surface in surfaces:
        x = 0
        for vertex in surface:
            x += 1
            glColor3fv(color)
            glVertex3fv(ground_surface[vertex])
    glEnd()


def Cylinder(center, radius, height, num_slices, Color=(0, 0, 0)):
    r = radius
    h = height
    n = float(num_slices)

    center = [center[0], -center[1]]

    circle_pts = []
    for i in range(int(n) + 1):
        angle = 2 * math.pi * (i / n)
        x = r * math.cos(angle) + center[1]
        y = r * math.sin(angle) + center[0]
        pt = (x, y)
        circle_pts.append(pt)

    glBegin(GL_TRIANGLE_FAN)  # drawing the back circle
    glColor(Color[0], Color[1], Color[2])
    # glVertex(0, h / 2.0, 0)
    for (x, y) in circle_pts:
        z = 0.0
        glVertex(x, z, y)
    glEnd()

    glBegin(GL_TRIANGLE_FAN)  # drawing the front circle
    glColor(Color[0], Color[1], Color[2])
    # glVertex(0, h / 2.0, 0)
    for (x, y) in circle_pts:
        z = h
        glVertex(x, z, y)
    glEnd()

    glBegin(GL_TRIANGLE_STRIP)  # draw the tube
    glColor(Color[0], Color[1], Color[2])
    for (x, y) in circle_pts:
        z = h
        glVertex(x, 0, y)
        glVertex(x, z, y)
    glEnd()


def Cone(center, radius, height, num_slices, Color=(0, 0, 0)):
    r = radius
    h = height
    n = float(num_slices)

    center = [center[0], -center[1]]

    circle_pts = []
    for i in range(int(n) + 1):
        angle = 2 * math.pi * (i / n)
        x = r * math.cos(angle) + center[1]
        y = r * math.sin(angle) + center[0]
        pt = (x, y)
        circle_pts.append(pt)

    glBegin(GL_TRIANGLE_FAN)  # drawing the back circle
    glColor(Color[0], Color[1], Color[2])
    # glVertex(0, h / 2.0, 0)
    for (x, y) in circle_pts:
        z = 0.0
        glVertex(x, z, y)
    glEnd()

    glBegin(GL_TRIANGLE_FAN)  # drawing the front circle
    glColor(Color[0], Color[1], Color[2])
    # glVertex(0, h / 2.0, 0)
    for (x, y) in circle_pts:
        z = h
        glVertex(center[1], z, center[0])
    glEnd()

    glBegin(GL_TRIANGLE_STRIP)  # draw the tube
    glColor(Color[0], Color[1], Color[2])
    for (x, y) in circle_pts:
        z = h
        glVertex(x, 0, y)
        glVertex(center[1], z, center[0])
    glEnd()


def wrapTo180(angle):
    angle_comp = angle.copy()
    if len(angle_comp) == 1:
        while angle_comp > 180:
            angle_comp -= 360
        while angle_comp < -180:
            angle_comp += 360
    else:
        while np.any(angle_comp > 180):
            angle_comp[angle_comp > 180] -= 360
        while np.any(angle_comp < -180):
            angle_comp[angle_comp < -180] += 360

    return angle_comp


class Agent_sim(pyglet.window.Window):
    def __init__(self, name_saved,
                 CX_Condition, MB_Condition,
                 Vin_K, MB_K, Oin_K,
                 Obj_names, Obj_position,
                 Lesion,
                 width, height,
                 *args, **kwargs):
        super(Agent_sim, self).__init__(width, height, *args, **kwargs)

        self.TreadmillMode = False

        self.lightfv = ctypes.c_float * 4

        self.dark = False

        self.CX_Condition = CX_Condition[0]
        self.MB_condition = MB_Condition
        self.nshift = CX_Condition[1]

        self.Obj_names = Obj_names
        self.Obj_position = Obj_position

        self.angle2Obj = np.arctan2(Obj_position[0][1], Obj_position[0][0])

        self.name_saved = name_saved
        self.Result_mat = []
        self.EPG_activity = []
        self.PEN_activity = []
        self.PEG_activity = []
        self.PFL_activity = []
        self.EN_activity = []
        self.CPU1_gain = []

        self.Width = width
        self.Height = height

        self.lesion = Lesion

        ## Feeder position on the arena edge
        angle_feeder = np.radians(0.0)
        self.position_feeder = [np.cos(angle_feeder), np.sin(angle_feeder)] * 100 # 100, 0

        error_angle = angle_feeder# + np.random.normal(0, 5, 1)
        distance_error = np.random.uniform(45, 55, 1)

        ## Agent initial position (xy) and height (z)
        self.translation_x = np.cos(np.radians(error_angle)) * distance_error
        self.translation_y = np.sin(np.radians(error_angle)) * distance_error
        self.translation_z = 0.0  # Height

        ## Agent initial orientation (z -> yaw | x&y -> pitch and roll fixed)
        self.rotation_x = 0.0
        self.rotation_y = 90.0
        self.rotation_z = np.arctan2((self.position_feeder[1] - self.translation_y),
                                     (self.position_feeder[0] - self.translation_x)) #orientation_to the feeder

        ## Constant movement step
        self.speed = 0.1 #0.25
        self.it = 0

        ## Pathway gains
        self.Vin_K = Vin_K
        self.Oin_K = Oin_K
        self.MB_K = MB_K

        ## Generate the Visual Units of size (20,20) from pixel size (self.Height, self.Width)
        self.Eye_rf = eye_generate2((20, 20), (self.Height, self.Width))
        self.Eye_rf_ang = np.zeros((len(self.Eye_rf[:, 0]), 2))
        self.Eye_rf_ang[:, 0] = (self.Eye_rf[:, 2] + self.Eye_rf[:, 3])/2
        self.Eye_rf_ang[:, 0] = -(self.Eye_rf_ang[:, 0]/max(self.Eye_rf[:, 3]) * 360 - 180)

        self.Eye_rf_ang[:, 1] = (self.Eye_rf[:, 0] + self.Eye_rf[:, 1])/2
        # ratio_HW = height/width
        ratio_HW = float(width / 4) / float(height)
        # print(ratio_HW)
        self.Eye_rf_ang[:, 1] = -(self.Eye_rf_ang[:, 1]/max(self.Eye_rf[:, 1]) * 360 - 180) * ratio_HW

        ## CX nets
        # Visual projection neurons" to the CX -> input (initialized)
        self.CX_pn = np.zeros((1, len(self.Eye_rf)))
        # Relative cardinal orientation
        self.CX_cards = (np.arange(8)+0.5)/8 * 360 - 180

        ## Build the transformation matrix CX_pn to EPGs
        self.CX_net = np.zeros((len(self.Eye_rf_ang), len(self.CX_cards)))
        Eye_horaim = np.tile(self.Eye_rf_ang[:, 0], (len(self.CX_cards), 1))
        Cards = np.transpose(np.tile(self.CX_cards, (len(self.Eye_rf_ang), 1)))
        shiftVin = 0.0#np.random.uniform(-180, 180, 1) # uncomment for random shift in the retinotopic connection scheme
        dist = abs(wrapTo180(Eye_horaim + shiftVin) - Cards)
        minimums = np.argmin(dist, axis=0)
        for iom in range(len(minimums)):
            self.CX_net[iom, minimums[iom]] = 1 #retinotopic

        ## Build the reward mask used for the innate attraction
        self.CX_Vin2Rew = self.Eye_rf_ang.copy()[:, 0]
        self.CX_Vin2Rew2 = self.Eye_rf_ang.copy()[:, 0]
        self.CX_Vin2Rew2 += self.angle2Obj
        self.CX_Vin2Rew2[self.CX_Vin2Rew2 > 180] -= 360
        self.CX_Vin2Rew2[self.CX_Vin2Rew2 < -180] += 360
        limitVisRew2 = 15.0
        limitVisRew = 15.0
        NoiseRatio = float(limitVisRew2/limitVisRew)
        self.CX_Vin2Rew2[abs(self.CX_Vin2Rew2) <= limitVisRew2] = 1.0/NoiseRatio
        self.CX_Vin2Rew2[abs(self.CX_Vin2Rew2) > limitVisRew2] = 0.0 #Pos
        # To shift the mask
        self.CX_Vin2Rew += 0 # shift value
        self.CX_Vin2Rew[self.CX_Vin2Rew > 180] -= 360
        self.CX_Vin2Rew[self.CX_Vin2Rew < -180] += 360
        ## Smooth version (uncomment the line to use)
        # self.CX_Vin2Rew = (90 - abs(self.CX_Vin2Rew)) / 180 #Pos&Neg
        # self.CX_Vin2Rew = (180 - abs(self.CX_Vin2Rew)) / 180 #Pos
        # self.CX_Vin2Rew = abs(self.CX_Vin2Rew) / 180 #Neg
        ## Discrete (uncomment the line to use)
        #Positive version
        self.CX_Vin2Rew[abs(self.CX_Vin2Rew) <= limitVisRew] = 1.0
        self.CX_Vin2Rew[abs(self.CX_Vin2Rew) > limitVisRew] = 0.0
        # Positive and Negative version
        # self.CX_Vin2Rew[abs(self.CX_Vin2Rew) <= limitVisRew] = 0.5
        # self.CX_Vin2Rew[abs(self.CX_Vin2Rew) >= 180 - limitVisRew] = -0.5
        # self.CX_Vin2Rew[(abs(self.CX_Vin2Rew) < 180 - limitVisRew) & (abs(self.CX_Vin2Rew) > limitVisRew)] = 0

        ## Randomized transformation matrix CX_pn to EPGs
        # dimNet = self.CX_net.shape
        # shuffledNet = self.CX_net.reshape(dimNet[0]*dimNet[1])
        # np.random.shuffle(shuffledNet)
        # self.CX_net = shuffledNet.reshape(dimNet) #randomized

        self.CX_tl_mem = np.zeros(len(self.CX_cards))

        ## Build the matrix of Delta7 network (EPGs inter-inhibitory pathway)
        self.CX_intercon = np.zeros((len(self.CX_cards), len(self.CX_cards)))
        for ineux in range(len(self.CX_cards)):
            for ineuy in range(len(self.CX_cards)):
                ## Homogenic inhibition
                if ineux == ineuy:
                    self.CX_intercon[ineux, ineuy] = 0.0
                else:
                    self.CX_intercon[ineux, ineuy] = -0.2

        # print('------')
        # print('Delta7')
        # print(self.CX_intercon)

        ## Build PEN to EPG connection matrix
        self.CX_PEN = np.zeros((1, len(self.CX_cards)*2))
        self.CX_PEN2EPG = np.zeros((len(self.CX_cards)*2, len(self.CX_cards)))
        for ineux in range(self.CX_PEN2EPG.shape[0]):
            for ineuy in range(self.CX_PEN2EPG.shape[1]):
                distNeu = ineux - ineuy
                if (ineux < len(self.CX_cards)) & ((distNeu == 1) | (distNeu == -7)):
                    self.CX_PEN2EPG[ineux, ineuy] = 1.0
                elif (ineux >= len(self.CX_cards)) & ((distNeu == 15) | (distNeu == 7)):
                    self.CX_PEN2EPG[ineux, ineuy] = 1.0

        # print('------')
        # print('PEN2EPG')
        # print(self.CX_PEN2EPG)

        ## Build EPG to PEN connection matrix
        self.CX_EPG2PEN = np.zeros((len(self.CX_cards), len(self.CX_cards)*2))
        for ineux in range(self.CX_EPG2PEN.shape[0]):
            for ineuy in range(self.CX_EPG2PEN.shape[1]):
                if ineux == ineuy:
                    self.CX_EPG2PEN[ineux, ineuy] = 1.0
                elif abs(ineux-ineuy) == 8:
                    self.CX_EPG2PEN[ineux, ineuy] = 1.0

        ## Build EPG to PFL connection matrix
        self.CX_EPG2PFL = np.zeros((len(self.CX_cards), len(self.CX_cards)*2))
        for ineux in range(self.CX_EPG2PFL.shape[0]):
            for ineuy in range(self.CX_EPG2PFL.shape[1]):
                if ineux == ineuy:
                    self.CX_EPG2PFL[ineux, ineuy] = 1.0
                elif abs(ineux-ineuy) == 8:
                    self.CX_EPG2PFL[ineux, ineuy] = 1.0

        ## Build EPG to FBn connection matrix
        self.CX_EPG2FBn = np.zeros((len(self.CX_cards), len(self.CX_cards)*2))
        for ineux in range(self.CX_EPG2FBn.shape[0]):
            for ineuy in range(self.CX_EPG2FBn.shape[1]):
                if ineux == ineuy:
                    self.CX_EPG2FBn[ineux, ineuy] = 1.0
                elif abs(ineux-ineuy) == 8:
                    self.CX_EPG2FBn[ineux, ineuy] = 1.0

        self.CX_FBn2PFL = np.zeros((len(self.CX_cards)*2, len(self.CX_cards)*2))
        for ineux in range(self.CX_FBn2PFL.shape[0]):
            for ineuy in range(self.CX_FBn2PFL.shape[1]):
                distNeu = ineuy - ineux
                if ineux < len(self.CX_cards):
                    if (ineuy < len(self.CX_cards)) & ((distNeu == 1) | (distNeu == -7)):
                        self.CX_FBn2PFL[ineux, ineuy] = 1.0
                if ineux >= len(self.CX_cards):
                    if (ineuy >= len(self.CX_cards)) & ((distNeu == -1) | (distNeu == 7)):
                        self.CX_FBn2PFL[ineux, ineuy] = 1.0

        ## Reverse shift from FBn to PFL
        # for ineux in range(self.CX_FBn2PFL.shape[0]):
        #     for ineuy in range(self.CX_FBn2PFL.shape[1]):
        #         distNeu = ineuy - ineux
        #         if ineux < len(self.CX_cards):
        #             if (ineuy < len(self.CX_cards)) & ((distNeu == -1) | (distNeu == 7)):
        #                 self.CX_FBn2PFL[ineux, ineuy] = 1.0
        #         if ineux >= len(self.CX_cards):
        #             if (ineuy >= len(self.CX_cards)) & ((distNeu == 1) | (distNeu == -7)):
        #                 self.CX_FBn2PFL[ineux, ineuy] = 1.0

        # print('----')
        # print('FBn2PFL')
        # print(self.CX_FBn2PFL)
        # print('#####')
        # print('')

        # print('------')
        # print('EPG2PEN')
        # print(self.CX_EPG2PEN)

        # print('------')
        # print('EPG2PFL')
        # print(self.CX_EPG2PFL)

        ## Build PEG to EPG connection matrix
        self.CX_PEG = np.zeros((1, len(self.CX_cards)*2))
        self.CX_PEG2EPG = np.zeros((len(self.CX_cards)*2, len(self.CX_cards)))
        for ineux in range(self.CX_PEG2EPG.shape[0]):
            for ineuy in range(self.CX_PEG2EPG.shape[1]):
                if ineux == ineuy:
                    self.CX_PEG2EPG[ineux, ineuy] = 1.0
                elif abs(ineux-ineuy) == 8:
                    self.CX_PEG2EPG[ineux, ineuy] = 1.0

        # print('------')
        # print('PEG2EPG')
        # print(self.CX_PEG2EPG)

        ## Build EPG to PEG connection matrix
        self.CX_EPG = np.zeros((1, len(self.CX_cards)))
        self.CX_EPG2PEG = np.zeros((len(self.CX_cards), len(self.CX_cards)*2))
        for ineux in range(self.CX_EPG2PEG.shape[0]):
            for ineuy in range(self.CX_EPG2PEG.shape[1]):
                if ineux == ineuy:
                    self.CX_EPG2PEG[ineux, ineuy] = 1.0
                elif abs(ineux-ineuy) == 8:
                    self.CX_EPG2PEG[ineux, ineuy] = 1.0

        # print('------')
        # print('EPG2PEG')
        # print(self.CX_EPG2PEG)

        self.CX_PFL = np.zeros((1, len(self.CX_cards)))


        ## Initialize the EPG to PFL synaptic weights
        # Version extracted from the Drosophila database (see also Rayshbuskyi et al.)
        self.CX_PFL3weightL_base = np.asarray([1.0, 0.867, 0.804, 0.993,
                                    0.685, 0.384, 0.307, 0.034])
        self.CX_PFL3weightR_base = np.asarray([0.064, 0.638, 0.677, 0.748,
                                    0.600, 0.838, 0.600, 1])
        # Arbitrarily set version (Left-Right symetrical heterogeneity)
        # self.CX_PFL3weightR_base = np.asarray([0.45, 0.5, 0.6, 0.6,
        #                                  0.55, 0.5, 0.45, 0.4])
        # self.CX_PFL3weightL_base = np.asarray([0.4, 0.45, 0.5, 0.55,
        #                                  0.6, 0.6, 0.5, 0.45])

        # Homogeneous version
        self.CX_PFL3weightR_flat = np.asarray([0.5, 0.5, 0.5, 0.5,
                                         0.5, 0.5, 0.5, 0.5])
        self.CX_PFL3weightL_flat = np.asarray([0.5, 0.5, 0.5, 0.5,
                                         0.5, 0.5, 0.5, 0.5])  # flat
        # Homogeneous and null version
        # self.CX_PFL3weightR_flat = np.asarray([0.0, 0.0, 0.0, 0.0,
        #                                  0.0, 0.0, 0.0, 0.0])
        # self.CX_PFL3weightL_flat = np.asarray([0.0, 0.0, 0.0, 0.0,
        #                                  0.0, 0.0, 0.0, 0.0])  # null

        ## Define the version used based on the parameter entry
        if (self.CX_Condition == 'wilson') | (self.CX_Condition == 'noCX'):
            self.CX_PFL3weight_R = self.CX_PFL3weightR_base
            self.CX_PFL3weight_L = self.CX_PFL3weightL_base

        elif self.CX_Condition == 'shiftRight':
            self.CX_PFL3weight_L = np.asarray(self.CX_PFL3weightL_base[self.nshift:].tolist()
                                       + self.CX_PFL3weightL_base[:self.nshift].tolist())
            self.CX_PFL3weight_R = np.asarray(self.CX_PFL3weightR_base[self.nshift:].tolist()
                                       + self.CX_PFL3weightR_base[:self.nshift].tolist()) #shiftedRight

        elif self.CX_Condition == 'shiftLeft':
            nshift = len(self.CX_PFL3weightL_base) - self.nshift

            self.CX_PFL3weight_L = np.asarray(self.CX_PFL3weightL_base[nshift:].tolist()
                                       + self.CX_PFL3weightL_base[:nshift].tolist())
            self.CX_PFL3weight_R = np.asarray(self.CX_PFL3weightR_base[nshift:].tolist()
                                       + self.CX_PFL3weightR_base[:nshift].tolist()) #shiftedLeft

        elif self.CX_Condition == 'flat':
            self.CX_PFL3weight_R = self.CX_PFL3weightR_flat.copy()
            self.CX_PFL3weight_L = self.CX_PFL3weightL_flat.copy()
            ## Randomly attributed weights (to try
            # self.CX_PFL3weight_L = np.random.uniform(0.2, 0.8, 8)
            # self.CX_PFL3weight_R = np.random.uniform(0.2, 0.8, 8)

        ## Set learning state to 0
        self.learning = False
        self.antilearning = False

        ## MB nets initiation
        self.ref_thresh = 0.5
        self.MB_learningrate = 0.2
        self.MB_pn = np.zeros((1, len(self.Eye_rf)))
        self.MBleft_kc = np.zeros((1, 10000))
        self.MBright_kc = np.zeros((1, 10000))
        self.MBleft_kc2en = np.ones((10000, 1))
        self.MBright_kc2en = np.ones((10000, 1))
        self.MBleft_kc2en_neg = np.ones((10000, 1))
        self.MBright_kc2en_neg = np.ones((10000, 1))
        self.MBleft_en = np.zeros(self.MBleft_kc2en.shape[1])
        self.MBright_en = np.zeros(self.MBright_kc2en.shape[1])
        self.MBleft_en_neg = np.zeros(self.MBleft_kc2en_neg.shape[1])
        self.MBright_en_neg = np.zeros(self.MBright_kc2en_neg.shape[1])

        ## Panoramic PN2KC
        self.MBright_net = generate_random_kc3(len(self.Eye_rf), self.MBright_kc.shape[1], min_pn=2, max_pn=5)
        self.MBleft_net = generate_random_kc3(len(self.Eye_rf), self.MBleft_kc.shape[1], min_pn=2, max_pn=5)
        ## no CrossOver
        # self.MBright_net = generate_random_kc3(int(len(self.Eye_rf)/2), self.MBright_kc.shape[1], min_pn=2, max_pn=5)
        # self.MBright_net = np.concatenate((np.zeros(self.MBright_net.shape), self.MBright_net))
        # self.MBleft_net = generate_random_kc3(int(len(self.Eye_rf)/2), self.MBleft_kc.shape[1], min_pn=2, max_pn=5)
        # self.MBleft_net = np.concatenate((self.MBleft_net, np.zeros(self.MBleft_net.shape)))
        ## CrossOver only
        # self.MBleft_net = generate_random_kc3(int(len(self.Eye_rf)/2), self.MBleft_kc.shape[1], min_pn=2, max_pn=5)
        # self.MBleft_net = np.concatenate((np.zeros(self.MBleft_net.shape), self.MBleft_net))
        # self.MBright_net = generate_random_kc3(int(len(self.Eye_rf)/2), self.MBright_kc.shape[1], min_pn=2, max_pn=5)
        # self.MBright_net = np.concatenate((self.MBright_net, np.zeros(self.MBright_net.shape)))

        ## For perfect image memory
        self.MBmemo = np.zeros(self.Eye_rf.shape[0])

        self.familiarity = 0.0
        self.proprioception = 0.0
        self.init = True
        self.learnCount = 0

        self.sensTurn = np.sign(np.random.normal(0, 1.0, 1))

        self.familiarity_mean = []

        ## Olfactory pathway: Creation of the source + gaussian
        self.source_dist2center = 150
        self.source_angularposition = 0.0
        self.source_position = [np.cos(self.source_angularposition) * self.source_dist2center,
                                np.sin(self.source_angularposition) * self.source_dist2center]
        dist2source = np.sqrt((self.translation_x - self.source_position[0])**2 +
                              (self.translation_y - self.source_position[1])**2)
        ## Direct olfactory input (concentration)
        self.Olfactory = 500000/(100 * np.sqrt(2*np.pi)) * np.exp(-0.5*(dist2source/100)**2)
        ## Differential input (initialized)
        self.OlfIN = 0.0

    def on_draw(self):
        self.clear()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLightfv(GL_LIGHT0, GL_POSITION, self.lightfv(0.0, 0.0, 0.0, 0.0))
        # glEnable(GL_LIGHT0)
        # glEnable(GL_LIGHTING)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_DEPTH_TEST)
        glShadeModel(GL_SMOOTH)

        nb_segments = 4

        # Angle_fov = 124
        Angle_fov = 132.5
        angle_adjust = -180 + 360/nb_segments/2#-157.5

        NEAR_CLIPPING_PLANE = 1
        FAR_CLIPPING_PLANE = 500

        ## Building the 4 povs (panoramic vision)
        for icam in range(nb_segments):

            glViewport(icam * round(self.Width / nb_segments), 0, round(self.Width / nb_segments), round(self.Height))
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            # gluPerspective(Angle_fov, float(self.Width / nb_segments) / float(self.Height), 0.05, 500.0)

            glFrustum(-NEAR_CLIPPING_PLANE * np.tan(np.radians(360/nb_segments/2)),
                      NEAR_CLIPPING_PLANE * np.tan(np.radians(360/nb_segments/2)),
                      0,
                      NEAR_CLIPPING_PLANE * np.tan(np.radians(360/nb_segments/2 * float(self.Width) / float(self.Height))),
                      NEAR_CLIPPING_PLANE, FAR_CLIPPING_PLANE)

            glMatrixMode(GL_MODELVIEW)  # // Select The Modelview Matrix
            glLoadIdentity()  # // Reset The Modelview Matrix
            glClear(GL_DEPTH_BUFFER_BIT)

            glRotated(0, 0, 1, 0)
            glRotated(0, 1, 0, 0)
            glRotated(0, 0, 0, 1)
            glRotated(self.rotation_z + 180 + angle_adjust, 0, 1, 0)
            # glRotated(90, 1, 0, 0)

            glTranslated(self.translation_y, self.translation_z, -self.translation_x)

            angle_adjust += 360/nb_segments

            if not self.dark:

                ## Simple world
                Cube(sky_color)
                Surface(sky_color)
                ## Building every object
                for iobj in range(len(self.Obj_names)):
                    if self.Obj_names[iobj] == 'Cylinder':
                        Cylinder(self.Obj_position[iobj], 52/2, 150, 1000)
                    elif self.Obj_names[iobj] == 'Cone':
                        Cone(self.Obj_position[iobj], 36/2, 120, 1000)
                    elif self.Obj_names[iobj] == 'Cube':
                        Cube2(self.Obj_position[iobj], 60, 60)

            else:
                Cube(sky_color)
                Surface(sky_color)
            # Cube2(position_cube, 20, 30)
            # Cylinder([math.cos(np.radians(0))*60, math.sin(np.radians(0))*60], 2/2, 10, 100, Color=(0, 1, 1))
            # visualization.draw(meshes)
            # self.label.draw()

    def MB_update(self, learning=False, antilearning=False):
        self.MB_pn = np.concatenate(self.vision)
        self.MBright_kc = self.MB_pn @ self.MBright_net
        self.MBright_kc[self.MBright_kc < self.ref_thresh] = 0
        self.MBright_kc[self.MBright_kc > self.ref_thresh] = 1
        self.MBleft_kc = self.MB_pn @ self.MBleft_net
        self.MBleft_kc[self.MBleft_kc < self.ref_thresh] = 0
        self.MBleft_kc[self.MBleft_kc > self.ref_thresh] = 1

        ##APL version (uncomment next block and comment previous block to use)
        # self.MB_pn = np.concatenate(self.vision)
        # self.MBright_kc = self.MB_pn @ self.MBright_net
        # MBright_kc_sorted = sorted(self.MBright_kc)
        # self.threshright_1 = MBright_kc_sorted[int(0.99*len(MBright_kc_sorted))]
        # self.MBright_kc[self.MBright_kc < self.threshright_1] = 0.0
        # self.MBright_kc[self.MBright_kc > self.threshright_1] = 1.0
        # self.MBleft_kc = self.MB_pn @ self.MBleft_net
        # MBleft_kc_sorted = sorted(self.MBleft_kc)
        # self.threshleft_1 = MBleft_kc_sorted[int(0.99*len(MBleft_kc_sorted))]
        # self.MBleft_kc[self.MBleft_kc < self.threshleft_1] = 0.0
        # self.MBleft_kc[self.MBleft_kc > self.threshleft_1] = 1.0

        ## KC to EN depression during learning (positive and negative [anti-])
        if learning:
            self.MBright_kc2en[self.MBright_kc == 1, 0] -= self.MB_learningrate
            self.MBright_kc2en[self.MBright_kc2en < 0] = 0
            self.MBleft_kc2en[self.MBleft_kc == 1, 0] -= self.MB_learningrate
            self.MBleft_kc2en[self.MBleft_kc2en < 0] = 0
        elif antilearning:
            self.MBright_kc2en_neg[self.MBright_kc == 1, 0] -= self.MB_learningrate
            self.MBright_kc2en_neg[self.MBright_kc2en_neg < 0] = 0
            self.MBleft_kc2en_neg[self.MBleft_kc == 1, 0] -= self.MB_learningrate
            self.MBleft_kc2en_neg[self.MBleft_kc2en_neg < 0] = 0

        if np.sum(self.MBright_kc) != 0:
            self.MBright_en = np.sum(self.MBright_kc @ self.MBright_kc2en)/np.sum(self.MBright_kc)
            self.MBright_en_neg = np.sum(self.MBright_kc @ self.MBright_kc2en_neg)/np.sum(self.MBright_kc)
        else:
            self.MBright_en = 0.0#np.sum(self.MBright_kc @ self.MBright_kc2en)
            self.MBright_en_neg = 0.0#np.sum(self.MBright_kc @ self.MBright_kc2en_neg)
        if np.sum(self.MBleft_kc) != 0:
            self.MBleft_en = np.sum(self.MBleft_kc @ self.MBleft_kc2en)/np.sum(self.MBleft_kc)
            self.MBleft_en_neg = np.sum(self.MBleft_kc @ self.MBleft_kc2en_neg)/np.sum(self.MBleft_kc)
        else:
            self.MBleft_en = 0.0#np.sum(self.MBleft_kc @ self.MBleft_kc2en)
            self.MBleft_en_neg = 0.0#np.sum(self.MBleft_kc @ self.MBleft_kc2en_neg)

        ## APL version (uncomment next block and comment previous block to use)
        # self.MBright_en = np.sum(self.MBright_kc @ self.MBright_kc2en)/(len(self.MBright_kc)*0.01) #APL
        # self.MBright_en_neg = np.sum(self.MBright_kc @ self.MBright_kc2en_neg)/(len(self.MBright_kc)*0.01) #APL
        # self.MBleft_en = np.sum(self.MBleft_kc @ self.MBleft_kc2en)/(len(self.MBleft_kc)*0.01) #APL
        # self.MBleft_en_neg = np.sum(self.MBleft_kc @ self.MBleft_kc2en_neg)/(len(self.MBleft_kc)*0.01) #APL

        ## Perfect image memory
        self.MBin = np.concatenate(self.vision)
        if learning:
            self.MBmemo += np.sign(self.MBin)

        return self.MBright_en, self.MBleft_en

    def CX_update(self):
        if np.max(self.vision) == 0:
            CX_Vin = np.concatenate(self.vision)
        else:
            CX_Vin = np.concatenate(self.vision/np.max(self.vision))

        ## Perfect compass
        Compass = wrapTo180(self.rotation_z)
        Compass_8Cards = 1 / abs(wrapTo180(Compass - self.CX_cards))
        Compass_8Cards = Compass_8Cards/np.max(Compass_8Cards)
        Compass_8Cards[Compass_8Cards > 1.0] = 1.0

        ## Reward innate attraction
        self.CX_rewardIN_Vin = 0
        self.CX_rewardIN_Vin = np.sum(self.vision * self.CX_Vin2Rew) * float(self.Vin_K)
        # print(self.CX_rewardIN)

        self.CX_rewardIN_Oin = 0
        if self.OlfIN > 0.0:
            self.CX_rewardIN_Oin = self.OlfIN * float(self.Oin_K)
        else:
            self.CX_rewardIN_Oin = 0.0

        ## Reward MB familiarity (reversal to 1 = familiar)
        self.CX_rewardIN_MB = 0

        if self.MB_condition == 'LesionRight':
            self.CX_rewardIN_MBright = 0
            self.CX_rewardIN_MBright_neg = 0
        else:
            self.CX_rewardIN_MBright = 1 - self.MBright_en
            self.CX_rewardIN_MBright_neg = 1 - self.MBright_en_neg

        if self.MB_condition == 'LesionLeft':
            self.CX_rewardIN_MBleft = 0
            self.CX_rewardIN_MBleft_neg = 0
        else:
            self.CX_rewardIN_MBleft = 1 - self.MBleft_en
            self.CX_rewardIN_MBleft_neg = 1 - self.MBleft_en_neg

        ## Cutout value < 0.25 (noise)
        if self.CX_rewardIN_MBright <= 0.25:
            self.CX_rewardIN_MBright = 0.0
        else:
            self.CX_rewardIN_MBright = self.CX_rewardIN_MBright * float(self.MB_K)
        if self.CX_rewardIN_MBright_neg <= 0.25:
            self.CX_rewardIN_MBright_neg = 0.0
        else:
            self.CX_rewardIN_MBright_neg = self.CX_rewardIN_MBright_neg * float(self.MB_K)

        if self.CX_rewardIN_MBleft <= 0.25:
            self.CX_rewardIN_MBleft = 0.0
        else:
            self.CX_rewardIN_MBleft = self.CX_rewardIN_MBleft * float(self.MB_K)
        if self.CX_rewardIN_MBleft_neg <= 0.25:
            self.CX_rewardIN_MBleft_neg = 0.0
        else:
            self.CX_rewardIN_MBleft_neg = self.CX_rewardIN_MBleft_neg * float(self.MB_K)

        ## Reward perfect image diff instead of MB model)
        # self.CX_rewardIN_MB = np.sum(CX_Vin * self.MBmemo)

        ## Sum Rewards
        self.CX_rewardIN_right = self.CX_rewardIN_Vin + self.CX_rewardIN_Oin + self.CX_rewardIN_MBright - self.CX_rewardIN_MBright_neg
        self.CX_rewardIN_left = self.CX_rewardIN_Vin + self.CX_rewardIN_Oin + self.CX_rewardIN_MBleft - self.CX_rewardIN_MBleft_neg

        ## Self motion input
        proprioception = self.proprioception
        rotRight = np.sign(np.max([-proprioception, 0]))
        rotLeft = np.sign(np.max([proprioception, 0]))
        CX_Min = np.concatenate((np.ones((1, 8))*rotLeft, np.ones((1, 8))*rotRight), axis=1)

        # if (np.sum(self.CX_EPG @ self.CX_EPG2PEN) + np.sum(CX_Min)) != 0:
        #     VisionFact = np.sum(self.CX_EPG @ self.CX_EPG2PEN)/(np.sum(self.CX_EPG @ self.CX_EPG2PEN) + np.sum(CX_Min))
        #     ProprioFact = 1 - VisionFact
        # else:
        #     VisionFact = 0.5
        #     ProprioFact = 0.5

        ## PEN and PEG activity level
        CX_PEN_temp = 0.75 * (self.CX_EPG @ self.CX_EPG2PEN) + 0.75 * CX_Min
        CX_PEN_temp[CX_PEN_temp < 0] = 0
        CX_PEN_temp[CX_PEN_temp > 1.0] = 1.0

        CX_PEG_temp = 1.0 * self.CX_EPG @ self.CX_EPG2PEG
        CX_PEG_temp[CX_PEG_temp < 0] = 0
        CX_PEG_temp[CX_PEG_temp > 1.0] = 1.0

        ## Visual input in the 8 octants
        Vin_quads = (CX_Vin @ self.CX_net)
        # Uncomment if normalized
        # if np.max(Vin_quads) != 0:
        #     Vin_quads = Vin_quads/(np.max(Vin_quads))

        ## EPG activity level
        CX_EPG_temp = (Vin_quads
                       + 2.5 * (self.CX_PEN @ self.CX_PEN2EPG)
                       + 1.0 * (self.CX_PEG @ self.CX_PEG2EPG))
        ## Uncomment to use the perfect compass
        # CX_EPG_temp = (Compass_8Cards
        #                + 2.5 * self.CX_PEN @ self.CX_PEN2EPG
        #                + 1.0 * self.CX_PEG @ self.CX_PEG2EPG)
        CX_EPG_temp[CX_EPG_temp < 0] = 0
        ## Inhibitory pathway is dealt sequentially to avoid unstability in the EB compass model
        ## make the compass model winner take all (almost)
        CX_EPG_temp += CX_EPG_temp @ self.CX_intercon
        CX_EPG_temp[CX_EPG_temp < 0] = 0
        CX_EPG_temp[CX_EPG_temp > 1.0] = 1.0

        ## FBn activity level (Self motion *5 to overtake completely)
        CX_FBn_temp = self.CX_EPG @ self.CX_EPG2FBn - CX_Min * 5.0
        # CX_CPU4_temp = 1/(1 + np.exp(-(a * (CX_CPU4_temp)) - b))
        CX_FBn_temp[CX_FBn_temp < 0.0] = 0.0
        CX_FBn_temp[CX_FBn_temp > 1.0] = 1.0

        ## FBn shift and integration of the reward signal
        CX_FBn_shifted = CX_FBn_temp @ self.CX_FBn2PFL

        CX_FBnR = np.asarray(CX_FBn_shifted[0, 0:8]) * self.CX_rewardIN_right
        CX_FBnL = np.asarray(CX_FBn_shifted[0, 8:]) * self.CX_rewardIN_left
        # print('FBN_L', CX_FBnL)

        ## Uncomment to deal with CX lesion
        # if self.lesion == 'None':
        #     pass
        # elif self.lesion == 'EPG':
        #     CX_EPG_temp = np.zeros(CX_EPG_temp.shape) * 0.1 + np.random.normal(0, 0.01, CX_EPG_temp.shape)
        #     CX_EPG_temp[CX_EPG_temp < 0] = 0
        # elif self.lesion == 'EPG_right':
        #     CX_EPG_temp[0:4] = np.zeros(CX_EPG_temp.shape) * 0.1 + np.random.normal(0, 0.01, CX_EPG_temp.shape)
        #     CX_EPG_temp[CX_EPG_temp < 0] = 0
        # elif self.lesion == 'EPG_left':
        #     CX_EPG_temp[4:] = np.zeros(CX_EPG_temp.shape) * 0.1 + np.random.normal(0, 0.01, CX_EPG_temp.shape)
        #     CX_EPG_temp[CX_EPG_temp < 0] = 0
        # elif self.lesion == 'PFL_left':
        #     # self.CX_EPG[0, 0:8] = np.ones(CX_EPG_temp.shape) * 0.1 + np.random.normal(0, 0.025, self.CX_EPG[0, 0:8].shape)
        #     self.CX_EPG[0, self.cell_silenced] = 0
        # elif self.lesion == 'PFL_right':
        #     # self.CX_EPG[0, 8:] = np.ones(CX_EPG_temp.shape) * 0.1 + np.random.normal(0, 0.025, self.CX_EPG[0, 8:].shape)
        #     self.CX_EPG[0, self.cell_silenced] = 0
        # elif self.lesion == 'PB_right':
        #     CX_PEN_temp[0, 0:8] = np.zeros(CX_PEN_temp[0, 0:8].shape) * 0.1 + np.random.normal(0, 0.1, CX_PEN_temp[0, 0:8].shape)
        #     CX_PEG_temp[0, 0:8] = np.zeros(CX_PEG_temp[0, 0:8].shape) * 0.1 + np.random.normal(0, 0.1, CX_PEG_temp[0, 0:8].shape)
        # elif self.lesion == 'PB_left':
        #     CX_PEN_temp[0, 8:] = np.zeros(CX_PEN_temp[0, 8:].shape) * 0.1 + np.random.normal(0, 0.1, CX_PEN_temp[0, 8:].shape)
        #     CX_PEG_temp[0, 8:] = np.zeros(CX_PEG_temp[0, 8:].shape) * 0.1 + np.random.normal(0, 0.1, CX_PEG_temp[0, 8:].shape)
        # elif self.lesion == 'PBPFL_right':
        #     self.CX_EPG[0, 0:8] = np.ones(CX_EPG_temp.shape) * 0.1 + np.random.normal(0, 0.025, self.CX_EPG[0, 0:8].shape)
        #     CX_PEN_temp[0, 0:8] = np.zeros(CX_PEN_temp[0, 0:8].shape) * 0.1 + np.random.normal(0, 0.1, CX_PEN_temp[0, 0:8].shape)
        #     CX_PEG_temp[0, 0:8] = np.zeros(CX_PEG_temp[0, 0:8].shape) * 0.1 + np.random.normal(0, 0.1, CX_PEG_temp[0, 0:8].shape)
        # elif self.lesion == 'PBPFL_left':
        #     self.CX_EPG[0, 8:] = np.ones(CX_EPG_temp.shape) * 0.1 + np.random.normal(0, 0.025, self.CX_EPG[0, 8:].shape)
        #     CX_PEN_temp[0, 8:] = np.zeros(CX_PEN_temp[0, 8:].shape) * 0.1 + np.random.normal(0, 0.1, CX_PEN_temp[0, 8:].shape)
        #     CX_PEG_temp[0, 8:] = np.zeros(CX_PEG_temp[0, 8:].shape) * 0.1 + np.random.normal(0, 0.1, CX_PEG_temp[0, 8:].shape)

        ## Modulation applied on the EPG-PFL synapses (gain to ensure stability - can be increased if the limitations in the gain are taken off)
        modulationR = np.asarray(CX_FBnR) * 0.001
        modulationL = np.asarray(CX_FBnL) * 0.001

        ## PFL weight update
        self.CX_PFL3weight_L += modulationL
        # print('modulationL', modulationL)
        self.CX_PFL3weight_R += modulationR
        ## uncomment for normalization (np.max can be replace by np.sum)
        # self.CX_PFL3weight_L = self.CX_PFL3weight_L/np.max(self.CX_PFL3weight_L)
        # self.CX_PFL3weight_R = self.CX_PFL3weight_R/np.max(self.CX_PFL3weight_R)

        ## Weights limits (works without)
        self.CX_PFL3weight_L[self.CX_PFL3weight_L > 0.8] = 0.8
        self.CX_PFL3weight_L[self.CX_PFL3weight_L < 0.2] = 0.2
        self.CX_PFL3weight_R[self.CX_PFL3weight_R > 0.8] = 0.8
        self.CX_PFL3weight_R[self.CX_PFL3weight_R < 0.2] = 0.2

        ## PFL neurons activity
        self.CX_PFLR = self.CX_EPG * self.CX_PFL3weight_L
        self.CX_PFLL = self.CX_EPG * self.CX_PFL3weight_R
        ## "Stone" version with FBn accumulating the memory (Multiplication is replace by a differernce)
        # self.CX_PFLR = np.asarray(self.CX_EPG[0, 0:8]).reshape((1, 8)) - self.CX_PFL3weight_R
        # self.CX_PFLL = np.asarray(self.CX_EPG[0, 8:]).reshape((1, 8)) - self.CX_PFL3weight_L
        self.CX_PFLR[self.CX_PFLR < 0.0] = 0.0
        self.CX_PFLL[self.CX_PFLL < 0.0] = 0.0
        
        ## Output neurons
        self.CX_hr = np.sum(self.CX_PFLR)
        self.CX_hl = np.sum(self.CX_PFLL)

        ## Difference Right-Left
        if self.CX_Condition == 'noCX':
            self.Delta_CX = 0.0 #noCX
        else:
            ## Reverse the difference if using the Stone version
            self.Delta_CX = (self.CX_hr - self.CX_hl) * 3.0# * (1.00 + np.random.normal(0, 0.1, 1))
            if self.Delta_CX > 2.5:
                self.Delta_CX = 2.5
            elif self.Delta_CX < -2.5:
                self.Delta_CX = -2.5

        self.CX_PEN = CX_PEN_temp.copy()
        self.CX_EPG = CX_EPG_temp.copy()
        self.CX_PFL = np.concatenate((self.CX_PFLL[0, :], self.CX_PFLR[0, :]))
        self.CX_PEG = CX_PEG_temp.copy()

        return self.Delta_CX

    def edge_detect(self, image):
        # Visual Unit activity calculation
        # Edge detection
        pn_out_float = np.zeros((1, len(self.Eye_rf)), dtype=np.float32)

        for i in range(len(self.Eye_rf)):

            zone = image[int(self.Eye_rf[i, 0]):int(self.Eye_rf[i, 1]),
                         int(self.Eye_rf[i, 2]):int(self.Eye_rf[i, 3])]

            sepinf = int(np.floor(zone.shape[1] / 2))
            sepsup = int(np.ceil(zone.shape[1] / 2))
            sepd = int(np.floor(zone.shape[0] / 2))
            sepg = int(np.ceil(zone.shape[0] / 2))
            zone1 = zone[:, 0:sepinf]
            zone2 = zone[:, sepsup:-1]
            zone3 = zone[0:sepd, :]
            zone4 = zone[sepg:-1, :]

            # Static detection
            contrast1 = np.mean(zone1.reshape(1, zone1.shape[0]*zone1.shape[1]))
            contrast2 = np.mean(zone2.reshape(1, zone2.shape[0]*zone2.shape[1]))
            contrast3 = np.mean(zone3.reshape(1, zone3.shape[0]*zone3.shape[1]))
            contrast4 = np.mean(zone4.reshape(1, zone4.shape[0]*zone4.shape[1]))
            static_edge_DG = abs((contrast1 - contrast2) / (contrast1 + contrast2 + 0.000001))
            static_edge_UD = abs((contrast3 - contrast4) / (contrast3 + contrast4 + 0.000001))
            pn_out_float[0, i] = (1.5 * static_edge_DG + 0.5 * static_edge_UD) / 2

        return pn_out_float

    def update(self, dt):
        t1 = time.time()

    #################
    #### Image buffer

        Posx, Posy, width, height = glGetIntegerv(GL_VIEWPORT)
        buff = glReadPixels(0, 0, Width, Height, GL_RGB, GL_UNSIGNED_BYTE)
        image_array = np.fromstring(buff, np.uint8)
        image = image_array.reshape(Height, Width, 3)
        image = 255 - image

    #############################
    #### Color channel separation

        image_red = image[:, :, 0]
        image_green = image[:, :, 1]
        image_blue = image[:, :, 2]
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        self.vision = self.edge_detect(np.flipud(image_gray))

        if self.init:
            ## Initiation
            position_relat = np.asarray(
                [self.position_feeder[0] - self.translation_x, self.position_feeder[1] - self.translation_y])
            orientation_relat = np.degrees(np.arctan2(position_relat[1], position_relat[0]))

            self.rotation_x = 0.0
            self.rotation_y = 0.0
            self.rotation_z = orientation_relat  # Yaw

            self.init = False
            self.learning = True
            # self.sens = np.sign(np.random.normal(0.0, 1.0, 1))

        elif self.learning:
            ## Learning phase
            angle_feeder = np.radians(0.0)
            self.position_feeder = [np.cos(angle_feeder), np.sin(angle_feeder)] * 100  # 100, 0

            error_angle = angle_feeder# + np.random.normal(0, 5, 1)
            if self.learnCount == 0:
                # distance_error = np.random.uniform(85, 95, 1)
                distance_error = np.random.uniform(45, 55, 1)
                self.translation_x = np.cos(np.radians(error_angle)) * distance_error
                self.translation_y = np.sin(np.radians(error_angle)) * distance_error
                self.translation_z = 0.0  # Height

            self.rotation_x = 0.0
            self.rotation_y = 0.0
            self.rotation_z = np.arctan2((self.position_feeder[1] - self.translation_y),
                                         (self.position_feeder[0] - self.translation_x)) + np.random.uniform(-15, 15, 1)  # orientation_relat # Yaw

            ## Un/Comment if (no) learning wished
            self.MB_update(learning=self.learning)

            self.learnCount += 1

            if self.learnCount == 100:
                self.learning = False
                self.antilearning = True
                self.learnCount = 0

                self.translation_x = 0.0
                self.translation_y = 0.0
                self.rotation_z = np.random.uniform(0, 360, 1)

        elif self.antilearning:
            angle_feeder = np.radians(0.0)
            self.position_feeder = [np.cos(angle_feeder), np.sin(angle_feeder)] * 100  # 100, 0

            error_angle = angle_feeder  # + np.random.normal(0, 5, 1)
            if self.learnCount == 0:
                # distance_error = np.random.uniform(85, 95, 1)
                distance_error = np.random.uniform(45, 55, 1)
                self.translation_x = np.cos(np.radians(error_angle)) * distance_error
                self.translation_y = np.sin(np.radians(error_angle)) * distance_error
                self.translation_z = 0.0  # Height

            self.rotation_x = 0.0
            self.rotation_y = 0.0
            self.rotation_z = np.arctan2((self.position_feeder[1] - self.translation_y),
                                         (self.position_feeder[0] - self.translation_x))\
                              + 180\
                              + np.random.uniform(-15, 15, 1)  # orientation_relat # Yaw

            ## Un/Comment if (no) learning wished
            # self.MB_update(antilearning=self.antilearning)

            self.learnCount += 1

            if self.learnCount == 100:
                self.learning = False
                self.antilearning = False
                self.translation_x = 0.0
                self.translation_y = 0.0
                self.rotation_z = np.random.uniform(0, 360, 1)

        else:
            self.it += 1

            Famright, Famleft = self.MB_update()
            ## Overall familiarity to record
            self.familiarity = (Famright + Famleft) / 2

            Delta_CX = self.CX_update()

            ## Steering noise level
            Noise_level = 10.0
            Delta_steering = Delta_CX + np.random.normal(0.0, Noise_level, 1)[0]
            self.rotation_z += Delta_steering
            self.proprioception = Delta_steering #+ np.random.normal(0, 5.0, 1)

            ## Treadmill mode keep the agent immobile
            if not self.TreadmillMode:
                self.translation_y += np.sin(np.radians(self.rotation_z)) * self.speed
                self.translation_x += np.cos(np.radians(self.rotation_z)) * self.speed

            ## Olfactory input calculation
            dist2source = np.sqrt((self.translation_x - self.source_position[0]) ** 2 +
                                  (self.translation_y - self.source_position[1]) ** 2)
            # Differential input
            self.OlfIN = (500000/(100 * np.sqrt(2*np.pi)) * np.exp(-0.5 * (dist2source / 100) ** 2)) - self.Olfactory
            # Direct input
            self.Olfactory = 500000/(100 * np.sqrt(2*np.pi)) * np.exp(-0.5 * (dist2source / 100) ** 2)

            ## Recordings
            self.Result_mat.append([float(np.round(self.translation_x, 4)),
                                    float(np.round(self.translation_y, 4)),
                                    float(np.round(self.rotation_z, 4)),
                                    float(np.round(self.familiarity, 4))])
            self.EPG_activity.append(self.CX_EPG[0, :].tolist())
            self.PEN_activity.append(self.CX_PEN[0, :].tolist())
            self.PEG_activity.append(self.CX_PEG[0, :].tolist())
            self.PFL_activity.append(self.CX_PFL.tolist())
            self.EN_activity.append([self.CX_hl, self.CX_hr])
            self.CPU1_gain.append(np.concatenate((self.CX_PFL3weight_L, self.CX_PFL3weight_R)).tolist())

            dist_center = np.sqrt(self.translation_x ** 2 + self.translation_y ** 2)

            if not self.TreadmillMode:
                if (dist_center > 100) | (self.it > 5000):
                    self.stop()
                # elif self.it > 2500:
                #     self.dark = False
                # elif self.it > 1500:
                #     self.dark = True
                # elif dist_center > 70:
                #     self.dark = True

                # if dist_center < 50:
                #     self.speed = 0.1
                # else:
                #     self.speed = 0.25
            else:
                if self.it > 3000:
                    self.stop()
                # elif self.it > 2000:
                #     self.dark = True


            # print('####')
            # print('')

    def stop(self):
        ## Stop & Save
        name_savedfile = self.name_saved + 'Results_XY.csv'
        a = np.asarray(self.Result_mat, dtype=np.float32)
        np.savetxt(name_savedfile, a, delimiter=",")
        name_savedfile = self.name_saved + 'EPG.csv'
        a = np.asarray(self.EPG_activity, dtype=np.float32)
        np.savetxt(name_savedfile, a, delimiter=",")
        name_savedfile = self.name_saved + 'PEN.csv'
        a = np.asarray(self.PEN_activity, dtype=np.float32)
        np.savetxt(name_savedfile, a, delimiter=",")
        name_savedfile = self.name_saved + 'PEG.csv'
        a = np.asarray(self.PEG_activity, dtype=np.float32)
        np.savetxt(name_savedfile, a, delimiter=",")
        name_savedfile = self.name_saved + 'PFL.csv'
        a = np.asarray(self.PFL_activity, dtype=np.float32)
        np.savetxt(name_savedfile, a, delimiter=",")
        name_savedfile = self.name_saved + 'EPG2PFLweights.csv'
        a = np.asarray(self.CPU1_gain, dtype=np.float32)
        np.savetxt(name_savedfile, a, delimiter=",")
        name_savedfile = self.name_saved + 'ENs.csv'
        a = np.asarray(self.EN_activity, dtype=np.float32)
        np.savetxt(name_savedfile, a, delimiter=",")

        self.close()


def eye_generate2(nb_pn_size, image_size):
    # print(image_size[0] / nb_pn_size[0])
    nb_pn_ver = int(image_size[0]/nb_pn_size[0])
    nb_pn_hor = int(image_size[1]/nb_pn_size[1])
    nb_pn = nb_pn_ver * nb_pn_hor
    receptive_fields = np.zeros([nb_pn, 4])
    cpt = 0
    for i in range(nb_pn_ver):
        for j in range(nb_pn_hor):
            limit_bottom = i*image_size[0]/nb_pn_ver
            limit_top = (i+1)*image_size[0]/nb_pn_ver+1
            limit_left = j*image_size[1]/nb_pn_hor
            limit_right = (j+1)*image_size[1]/nb_pn_hor+1
            receptive_fields[cpt, :] = [limit_bottom, limit_top, limit_left, limit_right]

            cpt += 1

    return receptive_fields


def generate_random_kc3(nb_pn, nb_kc, min_pn=10, max_pn=21, vicinity='off', pn_map=None,
                rnd=RNG, dtype=np.float32):
    """
    Create the synaptic weights among the Projection Neurons (PNs) and the Kenyon Cells (KCs).
    Choose the first sample that has dispersion below the baseline (early stopping), or the
    one with the lower dispersion (in case non of the samples' dispersion is less than the
    baseline).

    :param nb_pn:       the number of the Projection Neurons (PNs)
    :param nb_kc:       the number of the Kenyon Cells (KCs)
    :param min_pn:
    :param max_pn:
    :param aff_pn2kc:   the number of the PNs connected to every KC (usually 28-34)
                        if the number is less than or equal to zero it creates random values
                        for each KC in range [28, 34]
    :param nb_trials:   the number of trials in order to find a acceptable sample
    :param baseline:    distance between max-min number of projections per PN
    :param rnd:
    :type rnd: np.random.RandomState
    :param dtype:
    """

    pn2kc = np.zeros((nb_pn, nb_kc), dtype=dtype)
    if vicinity == 'off':
        for i in range(nb_kc):
            # nb_con = rnd.randint(min_pn, max_pn+1)
            nb_con = int(random.normalvariate(min_pn + (max_pn-min_pn)/2, (max_pn-min_pn)/10))
            vaff_pn2kc = rnd.permutation(nb_pn)
            pn_con = vaff_pn2kc[0:nb_con]

            pn2kc[pn_con, i] = 1
    elif vicinity == 'on':
        if pn_map is None:
            print('Error: No PN map specified')
            print('Cannot process vicinity model')
            quit()
        for i in range(nb_kc):
            nb_con = int(random.normalvariate(min_pn + (max_pn - min_pn) / 2, (max_pn - min_pn) / 10))
            central_con = rnd.permutation(nb_pn)[0]

            possib_con = central_con

    # if non of the samples have dispersion lower than the baseline,
    # return the less dispersed one
    return pn2kc


if __name__ == "__main__":
    ## Necessary parameter to create the agent object (Do NOT touch
    screensize = pyautogui.size()
    screensize = (round(screensize[0] / 2), round(screensize[1] / 2))
    Width = screensize[0]
    Height = screensize[1]

    ###########
    ## Name of the results folder to generate
    name_folder = 'CXMB_EllipsoidBody_FB2.0_MBdoubleMemo_Vin+MB_PosEPG2PFL_InvSteer_test/'
    # name_folder = 'CXMB_EllipsoidBody_Vin+MB_PosEPG2PFL_InvSteer/'
    # name_folder = 'CXMB_EllipsoidBody_FB1.0_FlyEM_Connectomic_shiftR2/'
    ## Check that the end of the folder name is / (Do NOT erase)
    if name_folder[-1] != '/':
        name_folder += '/'

    ## Select EPG-PFL3 initial state (only 1 line uncommented)
    # CX_Condition = ['normal', 0]
    CX_Condition = ['flat', 0]
    # CX_Condition = ['shiftLeft', 2] # second entry is the shift value
    # CX_Condition = ['shiftRight', 2]
    # CX_Condition = ['noCX', 0]

    ## Select CX with/without lesions (only 1 line uncommented)
    Lesion = 'None'
    # Lesion = 'PB_left'
    # Lesion = 'PB_right'
    # Lesion = 'EPG'
    # Lesion = 'EPG_left'
    # Lesion = 'EPG_right'
    # Lesion = 'PFL_left'
    # Lesion = 'PFL_right'
    # Lesion = 'PBPFL_right'
    # Lesion = 'MirroredSun'
    # Lesion = 'CovEye_left'
    # Lesion = 'CovEye_right'

    ## Select MB with/without lesions (only 1 line uncommented)
    MB_Condition = 'None'
    # MB_Condition = 'LesionRight'
    # MB_Condition = 'LesionLeft'

    ## Number of landmark used
    nb_objects = 1

    ## Number of simulations to run
    NB_exp = 10

    ## Initialise the saving folder
    print('Folder name: ' + name_folder)

    os.mkdir(name_folder)

    filetxt = open(name_folder + "steer_params.txt", "a")
    filetxt.write('Exp' + '\t'
                  + 'Vin_Kp' + '\t'
                  + 'MB_Kp' + '\t'
                  + 'Oin_Kp' + '\n')
    filetxt.close()

    ## Save a copy of the python code used for the simulation
    src = 'Simu3D_CXMB.py'
    dst = name_folder + 'SimuCode.py'
    shutil.copyfile(src, dst)
    filetxt_params = open(name_folder + "Obj_parameters_exp.txt", "a")
    filetxt_params.write('Exp' + '\t'
                         + 'Object' + '\t'
                         + 'X' + '\t'
                         + 'Y' + '\n')
    filetxt_params.close()

    for iexp in range(NB_exp):

        Obj_names = []
        Obj_position = []

        #############
        ## World object(s) creation
        for iobj in range(nb_objects):
            randobj = np.random.random_integers(1, 3, 1) #random selection of object (1-3)
            # angular_position = 60.0#np.random.random_integers(0, 10, 1)[0] * 18 * np.sign(np.random.normal(0.0, 1.0, 1)[0])
            dist_position = 150.0 #distance to the center (rho)
            angular_position = -45 #angular position (phi) - uncomment/comment with next line to select fixed/random orientation
            # angular_position = np.random.uniform(-180, 180, 1)
            if randobj == 1:
                Obj_names.append('Cylinder')
                position = [np.cos(np.radians(angular_position)) * dist_position,
                            np.sin(np.radians(angular_position)) * dist_position]
                Obj_position.append(position)
            elif randobj == 2:
                Obj_names.append('Cone')
                position = [np.cos(np.radians(angular_position)) * dist_position,
                            np.sin(np.radians(angular_position)) * dist_position]
                Obj_position.append(position)
            elif randobj == 3:
                Obj_names.append('Cube')
                position = [np.cos(np.radians(angular_position)) * dist_position,
                            np.sin(np.radians(angular_position)) * dist_position]
                Obj_position.append(position)

            ## Save the xy coordinates and the shape for each object
            filetxt_params = open(name_folder + "Obj_parameters_exp.txt", "a")
            filetxt_params.write(str(iexp) + '\t'
                                 + Obj_names[iobj] + '\t'
                                 + str(float(Obj_position[iobj][0])) + '\t'
                                 + str(float(Obj_position[iobj][1])) + '\n')

        filetxt_params.close()

        #############
        ## Sensory gain parameters
        #############
        ## Visual innate pathway
        Vin_K = np.random.uniform(0.5, 1.0, 1)[0]
        ## Olfactory pathway
        Oin_K = 0.0#np.random.uniform(0.8, 1.2, 1)[0]
        ## Mushroom body Long-term Memory pathway
        MB_K = np.random.uniform(3.5, 4.5, 1)[0]

        ## Record the parameters for each simulation
        filetxt = open(name_folder + "steer_params.txt", "a")
        filetxt.write(str(iexp+1).zfill(3) + '\t'
                      + str(Vin_K) + '\t'
                      + str(MB_K) + '\t'
                      + str(Oin_K) + '\n')
        filetxt.close()

        #############
        ## Create the agent object
        #############
        print('Experiment #' + str(iexp+1).zfill(3) + '/' + str(NB_exp))
        Agent = Agent_sim(name_folder + 'Exp' + str(iexp+1).zfill(3),
                          CX_Condition, MB_Condition,
                          Vin_K, MB_K, Oin_K,
                          Obj_names, Obj_position,
                          Lesion,
                          Width, Height, "My Window",
                          fullscreen=False, resizable=True)

        #############
        ## Launch the simulation
        #############
        pyglet.clock.schedule_interval(Agent.update, 1.0/500.0)
        pyglet.app.run()
        pyglet.clock.unschedule(Agent.update) #avoid interaction between simulation runs


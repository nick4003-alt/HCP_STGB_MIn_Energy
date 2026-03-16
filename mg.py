# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 21:41:59 2024

@author: lundb
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import shutil as sh
import sys 
import time
from scipy.spatial import cKDTree

plt.close('all')

def GenerateAngles(rotation_axis, a, c, nymax, mzmax):
    """
    Generates all possible tilt angles based on maximum value of nymax and mzmax
    """
    angles = np.zeros((nymax, mzmax))
    anglestemp = np.zeros((1, 3))

    if rotation_axis == '1-210':
        for i in range(1, nymax):
            for k in range(1, mzmax):
                angles[i, k] = np.rad2deg(np.arctan(k*c / (i*np.sqrt(3)*a)))
                anglestemp = np.append(anglestemp, np.array([[i, k, angles[i, k]]]), axis=0)
    elif rotation_axis == '0-110':
        for i in range(1, nymax):
            for k in range(1, mzmax):
                angles[i, k] = np.rad2deg(np.arctan(k*c / (i*a)))
                anglestemp = np.append(anglestemp, np.array([[i, k, angles[i, k]]]), axis=0)

    anglestemp = np.round(anglestemp, decimals=8)
    anglestemp = anglestemp[np.unique(anglestemp[:, [-1]], return_index=True, axis=0)[1]]
    anglestemp = np.delete(anglestemp, 0, 0)
    anglestemp = anglestemp[anglestemp[:, -1].argsort()]

    print("Total angles generated: ", len(anglestemp))
    return anglestemp

def RotateMatrix(supercell, theta):
    """
    Rotation of a matrix - vectorized, no loop
    """
    rotation_matrix = np.array(([1, 0, 0],
                                 [0, np.cos(theta), -np.sin(theta)],
                                 [0, np.sin(theta),  np.cos(theta)]))
    return (rotation_matrix @ supercell.T).T

def RotateVector(vector, theta):
    """
    Rotation of a vector
    """
    rotation_matrix = np.array(([1, 0, 0],
                                 [0, np.cos(theta), -np.sin(theta)],
                                 [0, np.sin(theta),  np.cos(theta)]))
    return rotation_matrix @ vector

def CreateTopPart(rotation_axis, a, c, nx, ny, nz):
    """
    Creates the top part of the cell - vectorized, no triple loop
    """
    xv, yv, zv = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
    xv = xv.ravel(); yv = yv.ravel(); zv = zv.ravel()

    if rotation_axis == '1-210':
        c0 = np.column_stack((xv*a,                                  yv*a*np.sqrt(3),                       zv*c        ))
        c1 = np.column_stack((a/2 + xv*a,                            a*np.sqrt(3)/2 + yv*a*np.sqrt(3),      zv*c        ))
        c2 = np.column_stack((xv*a,                                  a*3*np.sqrt(3)/4 + yv*a*np.sqrt(3),    c/2 + zv*c  ))
        c3 = np.column_stack((a/2 + xv*a,                            a*np.sqrt(3)/4 + yv*a*np.sqrt(3),      c/2 + zv*c  ))
    elif rotation_axis == '0-110':
        c0 = np.column_stack((xv*a*np.sqrt(3),                       yv*a,                                  zv*c        ))
        c1 = np.column_stack((a*np.sqrt(3)/2 + xv*a*np.sqrt(3),      a/2 + yv*a,                            zv*c        ))
        c2 = np.column_stack((a*np.sqrt(3)/4 + xv*a*np.sqrt(3),      yv*a,                                  c/2 + zv*c  ))
        c3 = np.column_stack((a*3*np.sqrt(3)/4 + xv*a*np.sqrt(3),    a/2 + yv*a,                            c/2 + zv*c  ))

    coords = np.empty((4*nx*ny*nz, 3))
    coords[0::4] = c0; coords[1::4] = c1; coords[2::4] = c2; coords[3::4] = c3
    return coords

def CreateBottomPart(rotation_axis, a, c, na, nb, nc):
    """
    Creates the bottom part of the cell - vectorized, no triple loop
    """
    xv, yv, zv = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
    xv = xv.ravel(); yv = yv.ravel(); zv = zv.ravel()

    if rotation_axis == '1-210':
        c0 = np.column_stack((xv*a,                                  yv*a*np.sqrt(3),                       -zv*c       ))
        c1 = np.column_stack((a/2 + xv*a,                            a*np.sqrt(3)/2 + yv*a*np.sqrt(3),      -zv*c       ))
        c2 = np.column_stack((xv*a,                                  a*3*np.sqrt(3)/4 + yv*a*np.sqrt(3),    -c/2 - zv*c ))
        c3 = np.column_stack((a/2 + xv*a,                            a*np.sqrt(3)/4 + yv*a*np.sqrt(3),      -c/2 - zv*c ))
    elif rotation_axis == '0-110':
        c0 = np.column_stack((xv*a*np.sqrt(3),                       yv*a,                                  -zv*c       ))
        c1 = np.column_stack((a*np.sqrt(3)/2 + xv*a*np.sqrt(3),      a/2 + yv*a,                            -zv*c       ))
        c2 = np.column_stack((a*np.sqrt(3)/4 + xv*a*np.sqrt(3),      yv*a,                                  -c/2 - zv*c ))
        c3 = np.column_stack((a*3*np.sqrt(3)/4 + xv*a*np.sqrt(3),    a/2 + yv*a,                            -c/2 - zv*c ))

    coords = np.empty((4*nx*ny*nz, 3))
    coords[0::4] = c0; coords[1::4] = c1; coords[2::4] = c2; coords[3::4] = c3
    return coords

def Shift(cell, shiftx, shifty, a, c, na, nc, h):
    """
    Shifts the atoms in both x and y direction - vectorized wrapping
    """
    print("shiftx: ", shiftx)
    print("shifty: ", shifty)

    ans = np.copy(cell)
    delta_y = np.sqrt((na*a*np.sqrt(3))**2 + (nc*c)**2)
    ans = ans + np.array([[h[0,0]*shiftx, delta_y*shifty, 0]])

    print("period-x: ", h[0,0])
    print("period-y: ", delta_y)
    print("shiftx_dist: ", h[0,0]*shiftx)
    print("shifty_dist: ", delta_y*shifty)

    # Vectorized wrap
    ans[:, 0] -= np.floor(ans[:, 0] / h[0,0]) * h[0,0]
    ans[:, 1] -= np.floor(ans[:, 1] / h[1,1]) * h[1,1]

    return ans

def WriteToLAMMPS(rotation_axis, supercell, h, path):

    xlo = 0;  xhi = h[0,0]
    ylo = 0;  yhi = h[1,1]
    zlo = -h[2,2]; zhi = h[2,2]

    f = open(path, "w")
    f.write("#LAMMPS written by hcp code \n \n")
    f.write("%i atoms \n \n" % (len(supercell)))
    f.write("1 atom types \n \n")
    f.write("%10.8f %10.8f xlo xhi \n" % (xlo, xhi))
    f.write("%10.8f %10.8f ylo yhi \n" % (ylo, yhi))
    f.write("%10.8f %10.8f zlo zhi \n \n" % (zlo, zhi))
    f.write("Atoms \n \n")
    for i in range(len(supercell)):
        f.write("%i 1 %10.8f %10.8f %10.8f \n" % (i+1, supercell[i,0],
                                                    supercell[i,1],
                                                    supercell[i,2]))
    f.close()

def WriteToPOSCAR(rotation_axis, supercell, h, filename):

    supercell = supercell + np.array([[0, 0, h[2,2]]])

    f = open(rotation_axis + '/poscar/' + filename + ".txt", "w")
    f.write("VASP written by hcp code \n")
    f.write("1 \n")
    f.write("%10.8f %10.8f %10.8f \n" % (h[0,0], 0, 0))
    f.write("%10.8f %10.8f %10.8f \n" % (0, h[1,1], 0))
    f.write("%10.8f %10.8f %10.8f \n" % (0, 0, 2*h[2,2]))
    f.write("Ti \n")
    f.write(str(len(supercell)) + " \n")
    f.write("Cartesian \n")
    for i in range(len(supercell)):
        f.write("%10.8f %10.8f %10.8f \n" % (supercell[i,0],
                                              supercell[i,1],
                                              supercell[i,2]))
    f.close()

def CreateGBTop(rotation_axis, supercell, a, c, na, nb, nc, theta):
    """
    Creates the top part of the grain boundary by rotating and moving atoms
    """
    R0 = np.copy(supercell)
    if rotation_axis == '1-210':
        h0 = np.array(([nb*a, 0,                              0],
                        [0,    na*np.sqrt(3)*a,                0],
                        [0,    na*np.sqrt(3)*a*np.tan(theta),  nc*c]))
    elif rotation_axis == '0-110':
        h0 = np.array(([nb*a*np.sqrt(3), 0,                   0],
                        [0,               na*a,                0],
                        [0,               na*a*np.tan(theta),  nc*c]))

    S0 = np.round(np.linalg.inv(h0) @ np.transpose(R0), decimals=8)
    for i in range(3):
        for j in range(np.shape(S0)[1]):
            if S0[i,j] < 0:
                S0[i,j] += abs(np.floor(S0[i,j]))
    R1 = np.transpose(h0 @ S0)
    R2 = RotateMatrix(R1, -theta)
    h1 = np.transpose(RotateMatrix(np.transpose(h0), -theta))
    h1[1,2] = 0
    S1 = np.linalg.inv(h1) @ np.transpose(R2)
    for i in range(1, 3):
        for j in range(np.shape(S1)[1]):
            if S1[i,j] > 1:
                S1[i,j] -= abs(np.floor(S1[i,j]))
    R3 = np.round(np.transpose(h1 @ S1), decimals=8)

    return R3, h1

def CreateGBBot(rotation_axis, supercell, a, c, na, nb, nc, theta):
    """
    Creates the bottom part of the grain boundary by rotating and moving atoms
    """
    R0 = np.copy(supercell)
    if rotation_axis == '1-210':
        h0 = np.array(([nb*a, 0,                               0],
                        [0,    na*np.sqrt(3)*a,                 0],
                        [0,    -na*np.sqrt(3)*a*np.tan(theta),  -nc*c]))
    elif rotation_axis == '0-110':
        h0 = np.array(([nb*a*np.sqrt(3), 0,                    0],
                        [0,               na*a,                 0],
                        [0,               -na*a*np.tan(theta),  -nc*c]))

    S0 = np.round(np.linalg.inv(h0) @ np.transpose(R0), decimals=8)
    for i in range(3):
        for j in range(np.shape(S0)[1]):
            if S0[i,j] < 0:
                S0[i,j] += abs(np.floor(S0[i,j]))
    R1 = np.transpose(h0 @ S0)
    R2 = RotateMatrix(R1, theta)
    h1 = np.transpose(RotateMatrix(np.transpose(h0), theta))
    h1[1,2] = 0
    S1 = np.linalg.inv(h1) @ np.transpose(R2)
    for i in range(1, 3):
        for j in range(np.shape(S1)[1]):
            if S1[i,j] > 1:
                S1[i,j] -= abs(np.floor(S1[i,j]))
    R3 = np.round(np.transpose(h1 @ S1), decimals=8)

    return R3, h1

def ReplaceAtomsZ(top, bot, h):
    """
    Finds atoms that are too close and replaces them - vectorized using KDTree,
    no nested for loops.
    """
    def _remove_close_pairs(A, B, z_boundary):
        changed = True
        while changed:
            changed = False
            tree = cKDTree(B)
            dists, idxs = tree.query(A, k=1)
            close = np.where(dists < min_distance)[0]
            if len(close) == 0:
                break
            del_B = []
            del_A = []
            used_B = set()
            for ai in close:
                bi = idxs[ai]
                if bi in used_B:
                    continue
                used_B.add(bi)
                mid = (A[ai] + B[bi]) / 2
                if mid[2] <= z_boundary:
                    A[ai] = mid
                    del_B.append(bi)
                else:
                    B[bi] = mid
                    del_A.append(ai)
            if del_B:
                B = np.delete(B, del_B, axis=0)
                changed = True
            if del_A:
                A = np.delete(A, del_A, axis=0)
                changed = True
        return A, B

    # Pass 1
    bot_shifted = bot + np.array([[0, 0, 2*h[2,2]]])
    top, bot_shifted = _remove_close_pairs(top, bot_shifted, z_boundary=h[2,2])
    bot = bot_shifted - np.array([[0, 0, 2*h[2,2]]])
    top = top - np.array([[0, 0, 2*h[2,2]]])

    # Pass 2
    top, bot = _remove_close_pairs(top, bot, z_boundary=-h[2,2])
    top = top + np.array([[0, 0, 2*h[2,2]]])

    # Pass 3
    top, bot = _remove_close_pairs(top, bot, z_boundary=0)

    return top, bot

def FindN(angle, x, y, z):
    """
    Finds the values of nx, ny, nz closest to the desired dimensions
    """
    na, nc = angle[0:2]
    theta = np.deg2rad(angle[2])

    x_start   = x
    y_start   = y * np.cos(theta)
    z_start   = z / np.cos(theta)

    nx_start  = x_start / a
    ny_start  = y_start / a
    nz_start  = z_start / c

    nx = int(np.ceil(nx_start))

    ny = int(np.ceil(ny_start / na) * na)
    if ny == 0:
        ny = int(na)

    nz = int(np.ceil(nz_start / nc) * nc)
    if nz == 0:
        nz = int(nc)

    return nx, ny, nz

def AddAtomsTop(supercell, h, angle):
    """
    Adds atoms that are missing in top of the cells.
    """
    na, nc, theta = angle
    ans = np.copy(supercell)

    index = [i for i in range(len(ans)) if ans[i, 2] == 0]

    for i in index:
        x = ans[i, 0]
        y = ans[i, 1] + h[2,2] * np.tan(np.deg2rad(theta))
        z = h[2,2]
        new_atom = np.array([[x, y, z]])
        if 0 <= y <= h[1,1]:
            ans = np.append(ans, new_atom, axis=0)
        k = 1
        while y >= 0:
            x = ans[i, 0]
            y = ans[i, 1] + h[2,2] * np.tan(np.deg2rad(theta)) - k*h[1,1]
            z = h[2,2]
            new_atom = np.array([[x, y, z]])
            if 0 <= y <= h[1,1]:
                ans = np.append(ans, new_atom, axis=0)
            k += 1
    return ans

def AddAtomsBot(supercell, h, angle):
    """
    Adds atoms that are missing in bottom of the cells.
    """
    na, nc, theta = angle
    delta_y = np.sqrt((na*a*np.sqrt(3))**2 + (nc*c)**2)
    ans = np.copy(supercell)

    index = [i for i in range(len(ans)) if ans[i, 2] == 0]

    for i in index:
        x = ans[i, 0]
        y = ans[i, 1] + h[2,2] * np.tan(np.deg2rad(theta))
        z = -h[2,2]
        new_atom = np.array([[x, y, z]])
        if 0 <= y <= h[1,1]:
            ans = np.append(ans, new_atom, axis=0)
        k = 1
        while y >= 0:
            x = ans[i, 0]
            y = ans[i, 1] + h[2,2] * np.tan(np.deg2rad(theta)) - k*h[1,1]
            z = -h[2,2]
            new_atom = np.array([[x, y, z]])
            if 0 <= y <= h[1,1]:
                ans = np.append(ans, new_atom, axis=0)
            k += 1
    return ans


#####################################################################
"""
USER INPUT
"""
a            = 3.2326231    # Lattice parameter (Å)
c            = 5.2368494    # Lattice parameter (Å)
x            = 10           ## Desired depth (Å)
y            = 10           ## Desired width (Å)
z            = 40           ## Desired height of each grain (Å)
nymax        = 10           ## Largest ny for angle generation
nzmax        = 10           ## Largest nz for angle generation
min_distance = 1.4          # Min allowed interatomic distance (Å)
rot_ax       = '1-210'      # '1-210' or '0-110'
file_format  = 'lammps'     # 'lammps' or 'poscar'
#####################################################################

full_program = True
if full_program == True:

    # Generate all unique angles from nymax x nzmax combinations
    angles = GenerateAngles(rot_ax, a, c, nymax, nzmax)
    num_angles = len(angles)

    HERE = os.getcwd()

    if not os.path.isdir(HERE + '/' + rot_ax):
        os.mkdir(HERE + '/' + rot_ax)

    if file_format == 'lammps':
        FOLD = str(HERE + '/' + rot_ax + '/lammps')
        g = open(HERE + '/' + rot_ax + '/angles.txt', 'w')
    elif file_format == 'poscar':
        FOLD = str(HERE + '/' + rot_ax + '/poscar')
        g = open(HERE + '/' + rot_ax + '/angles.txt', 'w')

    print(" WARNING: This will delete all folders and files in \n", FOLD, "\n Do you want to continue? [y/n]")
    answer = input()
    if answer == 'y':
        if os.path.isdir(FOLD):
            sh.rmtree(FOLD)
        os.mkdir(FOLD)
    else:
        sys.exit("Answer was something other than 'y'")

    start_time = time.time()

    for i in range(num_angles):
        FOLD = str(HERE + '/' + rot_ax + '/' + file_format)
        STR  = str(FOLD + '/' + file_format + '_' + str(i))

        if not os.path.exists(STR):
            os.mkdir(STR)

        # ── Build the base structure ONCE per angle ───────────────────────
        print('i = ', i)
        theta = np.deg2rad(angles[i, -1])
        nx, ny, nz = FindN(angles[i, :], x, y, z)
        print("theta: ", theta)
        print("nx: ", nx, "  ny: ", ny, "  nz: ", nz)

        na = int(angles[i, 0])
        nc = int(angles[i, 1])

        coords_top = CreateTopPart(rot_ax, a, c, nx, ny, nz)
        coords_bot = CreateBottomPart(rot_ax, a, c, nx, ny, nz)

        gb_top_base, h  = CreateGBTop(rot_ax, coords_top, a, c, ny, nx, nz, theta)
        gb_bot_base, h2 = CreateGBBot(rot_ax, coords_bot, a, c, ny, nx, nz, theta)

        gb_top_base = AddAtomsTop(gb_top_base, h, angles[i])
        gb_bot_base = AddAtomsBot(gb_bot_base, h, angles[i])
        # ─────────────────────────────────────────────────────────────────

        # Sequential folder counter 0-79 for each angle
        folder_index = 0

        for xx in range(4):
            for yy in range(20):
                STR  = str(FOLD + '/' + file_format + '_' + str(i) + '/' + str(folder_index))

                if not os.path.exists(STR):
                    os.mkdir(STR)

                FILE = str(STR + '/coords.fulldata')

                print('i = ', i, '  folder = ', folder_index)

                # Fresh copies of the base structure — only shift varies
                gb_top = np.copy(gb_top_base)
                gb_bot = np.copy(gb_bot_base)

                gb_top = Shift(gb_top, xx/4, yy/20, a, c, na, nc, h)
                gb_top, gb_bot = ReplaceAtomsZ(gb_top, gb_bot, h)

                full_gb = np.concatenate((gb_top, gb_bot))
                print("Antal atomer: ", len(full_gb))

                if file_format == 'lammps':
                    WriteToLAMMPS(rot_ax, full_gb, h, FILE)
                elif file_format == 'poscar':
                    WriteToPOSCAR(rot_ax, full_gb, h, 'poscar_' + str(i))

                folder_index += 1

        # Write angle info once per angle
        g.write("%2i %11.8f\n" % (i, angles[i, 2]))

    g.close()
    end_time = time.time()
    print('Time: ', end_time - start_time)

from matplotlib import pyplot as plt
import numpy as np
import cv2
import os

def estimate_homographies(path, grid_size, square_size):
    
    img = os.path.basename(path).split('.')[0]
    image = cv2.imread(path)

    return_value, corners = cv2.findChessboardCorners(image, patternSize=grid_size)

    if return_value is False:
        print(f'{img} corners not found')
        return False, 0, 0, 0

    corners=corners.reshape((grid_size[0]*grid_size[1],2)).copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 100, 0.001)
    cv2.cornerSubPix(gray,corners,(5,5),(-1,-1),criteria)

    real_coordinates = np.empty_like(corners)

    for index, corner in enumerate(corners):

        grid_size_cv2 = tuple(reversed(grid_size))
        u_index, v_index = np.unravel_index(index, grid_size_cv2)

        x_mm = (u_index) * square_size
        y_mm = (v_index) * square_size

        real_coordinates[index,:] = [x_mm, y_mm]

    A = np.empty((0, 9), dtype=float)

    for index, corner in enumerate(corners):
        Xpixel = corners[index, 0]
        Ypixel = corners[index, 1]
        Xmm = real_coordinates[index, 0]
        Ymm = real_coordinates[index, 1]

        m = np.array([Xmm, Ymm, 1]).reshape(1, 3)
        O = np.array([0, 0, 0]).reshape(1, 3)

        A = np.vstack((A, np.hstack((m, O, -Xpixel * m))))
        A = np.vstack((A, np.hstack((O, m, -Ypixel * m))))

    U, S, Vtransposed = np.linalg.svd(A)

    h=Vtransposed.transpose()[:,-1]
    H = h.reshape(3, 3) 

    return True, H, corners, real_coordinates

def compute_K(HH):

    V = np.empty((0, 6), dtype=float)

    for HI in HH:
        V = np.vstack((V, np.hstack(vij_matrix(HI,0,1))))
        V = np.vstack((V, np.hstack(vij_matrix(HI,0,0)- vij_matrix(HI,1,1))))


    U, sigma, Stransposed = np.linalg.svd(V)

    b = Stransposed.transpose()[:,-1]

    B=[[b[0], b[1], b[3]],
       [b[1], b[2], b[4]],
       [b[3], b[4], b[5]]]

    B = np.array(B)

    if np.all(np.linalg.eigvals(B) > 0):
        L = np.linalg.cholesky(B)

    elif np.all(np.linalg.eigvals(B) < 0):
        L = np.linalg.cholesky(-B)

    else:
        print("error in the calculation of Cholesky factorization")

    K = np.linalg.inv(L.transpose())
    K = K/K[2,2]

    return K

def vij_matrix(H, i, j):

    vij = np.array([H[0,i]*H[0,j],
                    H[0,i]*H[1,j] + H[1,i]*H[0,j],
                    H[1,i]*H[1,j],
                    H[2,i]*H[0,j] + H[0,i]*H[2,j],
                    H[2,i]*H[1,j] + H[1,i]*H[2,j],
                    H[2,i]*H[2,j]])
    
    vij = vij.transpose()

    return vij

def compute_Rt(H, K):

    lambd = 1/(np.linalg.norm(np.linalg.inv(K)@ H[:,0]))

    r1 = lambd*np.linalg.inv(K)@H[:,0]
    r2 = lambd*np.linalg.inv(K)@H[:,1]
    r3 = np.cross(r1, r2)
    R = np.column_stack((r1, r2, r3))
    
    t = lambd*np.linalg.inv(K)@H[:,2]
    t= t.reshape(3, 1)

    if not np.allclose(R.transpose(), np.linalg.inv(R)):
        U, S, Vtransposed = np.linalg.svd(R)
        Rprime = U@Vtransposed
        R = Rprime

    return R, t

def compute_error(image):

    corners = image["corners"]
    real_coordinates = image["real_coordinates"]
    P = image["P"]

    error = 0
    for i, point in enumerate(corners):

        m = np.array([real_coordinates[i, 0], real_coordinates[i, 1], 0, 1])
        error = error + ((P[0, :]@m)/(P[2, :]@m) - point[0])**2 + ((P[1, :]@m)/(P[2, :]@m) - point[1])**2

    error=round(error, 2)
    image["error"] = error
    return error

def draw_points(figure, image):

    corners = image["corners"]
    real_coordinates = image["real_coordinates"]
    P = image["P"]

    for i, point in enumerate(corners):

        m = np.array([real_coordinates[i, 0], real_coordinates[i, 1], 0, 1])
        cv2.circle(figure, (round(point[0]),round(point[1])), 5, color=(0,0,255), thickness=1)
        cv2.circle(figure, (round((P[0, :]@m)/(P[2, :]@m) ),round((P[1, :]@m)/(P[2, :]@m))), 5, color=(255,0,0), thickness=1)
    
    plt.figure(num = f'{image["name"]}', figsize = (6.4*2,4.8*2))
    plt.axis('off')
    plt.imshow(figure)

    text = f'R={np.round(image["R"],2)}\n'
    text += f't={np.round(image["t"].transpose(),3)}'
    text += '$^{T}$\n'
    text += f'error={image["error"]}'
    
    t = plt.text(.01, .99, text, ha='left', va='top', color='black')
    t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='white'))

    if image["personal"] is True:
        path = f'results/personal'
    else:
        path = f'results'

    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(f'{path}/{image["name"]} - points.png', bbox_inches='tight', pad_inches = 0)
    plt.show()

def superimpose_cylinder(figure, image, a, b, r, h):

    P = image["P"]
    tz = image["t"][2]

    stepSize = 0.1
    num_steps = int(2 * np.pi / stepSize)
    positionsX = np.zeros(num_steps)
    positionsY = np.zeros(num_steps)
    positionsZ = np.zeros(num_steps)

    t = 0
    for i in range(num_steps):
        positionsX[i] = r * np.cos(t) + a
        positionsY[i] = r * np.sin(t) + b
        positionsZ[i] = 0
        t += stepSize
        
    circle_inf = P @ np.vstack((positionsX, positionsY, positionsZ, np.ones_like((positionsX))))
    if tz >= 0:
        circle_sup = P @ np.vstack((positionsX, positionsY, positionsZ + h, np.ones_like((positionsX))))
    else:
        circle_sup = P @ np.vstack((positionsX, positionsY, positionsZ - h, np.ones_like((positionsX))))


    circle_inf = (circle_inf/circle_inf[2])[:2]
    circle_sup = (circle_sup/circle_sup[2])[:2]

    overlay = figure.copy()

    cv2.polylines(figure, np.int32([circle_inf.transpose()]), isClosed=True, color=(0, 0, 0), thickness=2)
    cv2.fillPoly(overlay, np.int32([circle_inf.transpose()]), color=(255, 0, 0))

    alpha = 0.7
    figure = cv2.addWeighted(overlay, alpha, figure, 1 - alpha, 0)

    overlay = figure.copy()

    cv2.polylines(figure, np.int32([circle_sup.transpose()]), isClosed=True, color=(0, 0, 0), thickness=2)
    cv2.fillPoly(overlay, np.int32([circle_sup.transpose()]), color=(0, 255, 0))

    figure = cv2.addWeighted(overlay, alpha, figure, 1 - alpha, 0)
    
    alpha = 0.2
    for i in range(h):
        overlay = figure.copy()
        if tz >= 0:
            circle_int = P @ np.vstack((positionsX, positionsY, positionsZ + i, np.ones_like((positionsX))))
        else:
            circle_int = P @ np.vstack((positionsX, positionsY, positionsZ - i, np.ones_like((positionsX))))

        circle_int = (circle_int/circle_int[2])[:2]
        cv2.polylines(overlay, np.int32([circle_int.transpose()]), isClosed=True, color=(128, 128, 128), thickness=1)
        figure = cv2.addWeighted(overlay, alpha, figure, 1 - alpha, 0)

    plt.figure(num = f'{image["name"]} - cylinder', figsize = (6.4*2,4.8*2))
    plt.axis('off')
    plt.imshow(figure)

    if image["personal"] is True:
        path = f'results/personal'
    else:
        path = f'results'

    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(f'{path}/{image["name"]} - cylinder.png', bbox_inches='tight', pad_inches = 0)
    plt.show()

    return 0


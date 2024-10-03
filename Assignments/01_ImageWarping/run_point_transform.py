import cv2
import numpy as np
import gradio as gr

# 初始化全局变量，存储控制点和目标点
points_src = []
points_dst = []
image = None

# 上传图像时清空控制点和目标点
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()  # 清空控制点
    points_dst.clear()  # 清空目标点
    image = img
    return img

# 记录点击点事件，并标记点在图像上，同时在成对的点间画箭头
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]  # 获取点击的坐标
    
    # 判断奇偶次来分别记录控制点和目标点
    if len(points_src) == len(points_dst):
        points_src.append([x, y])  # 奇数次点击为控制点
    else:
        points_dst.append([x, y])  # 偶数次点击为目标点
    
    # 在图像上标记点（蓝色：控制点，红色：目标点），并画箭头
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # 蓝色表示控制点
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # 红色表示目标点
    
    # 画出箭头，表示从控制点到目标点的映射
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)  # 绿色箭头表示映射
    
    return marked_image

# 执行仿射变换

def RBF(r, sigma):
    return np.exp(-(r*r)/(sigma**2))

def warp_RBF(image, sources_pts, wx, wy, sigma):
    # image warping by RBF
    n = sources_pts.shape[0]
    h = image.shape[0]
    w = image.shape[1]

    warped_image = np.zeros([h,w,3])
    M_Affine = np.array([wx[n:n+3,0], wy[n:n+3,0]]).T
    pts = np.array([[[1,i,j] for i in range(h)] for j in range(w)]).reshape([h*w,3])
    pts_affined = np.dot(pts, M_Affine) # shape (h*w, 2)
    
    pts_src = np.array([[[i,j] for i in range(h)] for j in range(w)]).reshape([h*w,2])
    r_ijk = pts_src.reshape([h*w, 1, 2]) - sources_pts.reshape([1, n, 2]) # shape (h*w, n, 2)
    r2_ijk = (r_ijk*r_ijk).sum(axis=2) # (h*w, n)
    rbf_ijk = RBF(np.sqrt(r2_ijk), sigma) # (h*w, n)
    w_ = np.concatenate((wx[0:n], wy[0:n]),axis=1)#.reshape([n, 2])
    pts_rbf = np.dot(rbf_ijk, w_) # (h*w, 2)

#    filled = np.zeros([h, w])
#    pts_dst = np.round(pts_affined + pts_rbf) # (h*w, 2)
#    pts_src_dst = np.concatenate((pts_src, pts_dst), axis=1) # (h*w, 4)
#    pts_src_dst = pts_src_dst.astype(np.uint)

#    x = pts_src_dst[:, 0]
#    y = pts_src_dst[:, 1]
#    u = pts_src_dst[:, 2]
#    v = pts_src_dst[:, 3]
#    valid_mask = (u >= 0) & (u < h) & (v >= 0) & (v < w)

#    valid_u = u[valid_mask]
#    valid_v = v[valid_mask]
#    valid_x = x[valid_mask]
#    valid_y = y[valid_mask]

#    warped_image[valid_u, valid_v] = image[valid_x, valid_y]
#    filled[valid_u, valid_v] = 1

    # resampling
#    warped_image = fill_img(warped_image, filled)

    # remap
    pts_dst = np.float32(pts_affined + pts_rbf).reshape([w,h,2]) # (h*w, 2)
    warped_image = cv2.remap(image, pts_dst[:,:,0], pts_dst[:,:,1], cv2.INTER_LINEAR)

    return warped_image.astype(np.uint8)

def fill_img(image, filled):
    # 使用 NumPy 切片获取形状
    image_filled = image.copy()  # 创建图像副本
    h, w = filled.shape
    
    # 创建掩码
    mask = (filled[1:h, 1:w] == 0)
    
    # 获取需要填充的坐标
    i_indices, j_indices = np.indices(mask.shape)
    
    # 计算填充像素值
    image_filled[1:h, 1:w][mask] = np.round(
        (image[0:h-1, 1:w][mask] / 2) + (image[1:h, 0:w-1][mask] / 2)
    )
    
    return image_filled

def fix_axis(coords):
    n = coords.shape[0]
    c = np.zeros([n,2])
    c[:,0] = coords[:,1]
    c[:,1] = coords[:,0]
    return c

def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """ 
    Return
    ------
        A deformed image.
    """
    
    warped_image = np.array(image)
    ### FILL: 基于MLS or RBF 实现 image warping

    n = source_pts.shape[0]
    image = np.array(image)
    tmp = target_pts
    target_pts = source_pts
    source_pts = tmp
    # source_pts, target_pts: n*2 arrays
    #source_pts = fix_axis(source_pts)
    #target_pts = fix_axis(target_pts)

    # 基于 RBF: Gaussian
    M = np.zeros([n+3, n+3])
    # G = (g||x_i-x_j||)_ij
    R = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            r2_ij = (source_pts[i,0]-source_pts[j,0])**2 \
                + (source_pts[i,1]-source_pts[j,1])**2
            R[i][j] = np.sqrt(r2_ij)
    sigma = (float(image.shape[0]+image.shape[1])) / 12
    print(image.shape)
    print("sigma = ", end='')
    print(sigma)
    G = RBF(R, sigma)
    print(G)

    M[0:n, 0:n] = G

    # Ve = [[1, x_j, y_j],...]
    Ve = np.zeros([3, n])
    Ve[0, :] = np.ones([1, n])
    Ve[1, :] = source_pts[:, 0]
    Ve[2, :] = source_pts[:, 1]

    M[0:n, n:n+3] = Ve.T
    M[n:n+3, 0:n] = Ve
    print("M =")
    print(M)

    # U, V
    U = np.zeros([n+3, 1])
    V = np.zeros([n+3, 1])
    U[0:n, 0] = target_pts[:, 0]
    V[0:n, 0] = target_pts[:, 1]

    X = np.linalg.solve(M, U)
    Y = np.linalg.solve(M, V)
    print("X=")
    print(X.T)
    print("Y=")
    print(Y.T)

    warped_image = warp_RBF(image, source_pts, X, Y, sigma)
    return warped_image

def run_warping():
    global points_src, points_dst, image ### fetch global variables

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# 清除选中点
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image  # 返回未标记的原图

# 使用 Gradio 构建界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source="upload", label="上传图片", interactive=True, width=800, height=200)
            point_select = gr.Image(label="点击选择控制点和目标点", interactive=True, width=800, height=800)
            
        with gr.Column():
            result_image = gr.Image(label="变换结果", width=800, height=400)
    
    # 按钮
    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")  # 添加清除按钮
    
    # 上传图像的交互
    input_image.upload(upload_image, input_image, point_select)
    # 选择点的交互，点选后刷新图像
    point_select.select(record_points, None, point_select)
    # 点击运行 warping 按钮，计算并显示变换后的图像
    run_button.click(run_warping, None, result_image)
    # 点击清除按钮，清空所有已选择的点
    clear_button.click(clear_points, None, point_select)
    
# 启动 Gradio 应用
demo.launch()

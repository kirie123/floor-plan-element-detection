import torch
def focal_loss(pred, target, alpha=2, beta=4):
    """Focal Loss for heatmap"""
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()

    neg_weights = torch.pow(1 - target, beta)
    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)

    pos_loss = torch.log(pred) * torch.pow(1 - pred, alpha) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss


def reg_l1_loss(pred, target, mask):
    """L1 regression loss"""
    return (torch.abs(pred - target) * mask).sum() / (mask.sum() + 1e-6)


def detection_loss(outputs, targets, num_classes=4, stride=16, img_size=1024, viz_dir = "heatmap_viz"):
    """
    CenterNet style loss
    """
    debug = False
    B, _, H, W = outputs["heatmap"].shape
    device = outputs["heatmap"].device
    if debug:
        print(f"\n[损失调试] 批次大小: {B}, 特征图尺寸: {H}x{W}")
        import os
        os.makedirs(viz_dir, exist_ok=True)
        import matplotlib.pyplot as plt
    # 特征图尺寸
    feat_size = img_size / stride  # 1024/16 = 64
    # 初始化目标张量
    hm_target = torch.zeros((B, num_classes, H, W), device=device)
    wh_target = torch.zeros((B, 2, H, W), device=device)
    offset_target = torch.zeros((B, 2, H, W), device=device)
    mask = torch.zeros((B, H, W), device=device, dtype=torch.bool)
    total_objects = 0
    assigned_objects = 0
    # 新增：收集宽高信息用于诊断
    wh_targets_list = []
    wh_preds_list = []
    for b in range(B):
        boxes = targets[b]["boxes"]  # 现在已经是归一化坐标 [N, 4]
        labels = targets[b]["labels"]  # [N]
        if debug:
            print(f"[损失调试] 批次 {b}: {len(boxes)} 个目标")

        total_objects += len(boxes)
        orig_size = targets[b]["orig_size"]  # [W, H]
        # 确保boxes和labels在正确的设备上
        boxes = boxes.to(device)
        labels = labels.to(device)
        orig_size = orig_size.to(device)
        for i in range(len(boxes)):
            # 反归一化到特征图尺度
            x1, y1, x2, y2 = boxes[i] * feat_size  # 从[0,1]映射到特征图尺度[0,64]
            cls_id = labels[i].item()

            # 计算中心点
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            cx_int, cy_int = int(cx), int(cy)

            # 确保在特征图范围内
            if 0 <= cx_int < W and 0 <= cy_int < H:
                assigned_objects += 1
                # # 热力图目标（高斯核）
                # radius = gaussian_radius((y2 - y1) , (x2 - x1) )
                # radius = max(0, int(radius))
                # draw_gaussian(hm_target[b, cls_id], (cx_int, cy_int), radius, device)
                # 热力图目标（椭圆高斯核）
                h, w = (y2 - y1).item(), (x2 - x1).item()
                sigma_x = max(w, 1.0) / 6.0  # 可根据效果调整分母，如4.0, 6.0, 8.0
                sigma_y = max(h, 1.0) / 6.0
                draw_elliptical_gaussian(hm_target[b, cls_id], (cx_int, cy_int), sigma_x, sigma_y, device)

                # 宽高目标
                w_feat = torch.log((x2 - x1) + 1e-6)
                h_feat = torch.log((y2 - y1) + 1e-6)
                # if cls_id == 0:# WALL
                #     print(f"墙宽高： {w_feat} , {h_feat}")
                wh_target[b, 0, cy_int, cx_int] = w_feat
                wh_target[b, 1, cy_int, cx_int] = h_feat

                # 偏移目标
                offset_x =  cx - cx_int
                offset_y = cy - cy_int
                offset_target[b, 0, cy_int, cx_int] = offset_x
                offset_target[b, 1, cy_int, cx_int] = offset_y

                mask[b, cy_int, cx_int] = True

                # 新增：收集宽高信息
                wh_targets_list.append((w_feat, h_feat))
                w_pred = outputs['wh'][b, 0, cy_int, cx_int].item()
                h_pred = outputs['wh'][b, 1, cy_int, cx_int].item()
                wh_preds_list.append((w_pred, h_pred))

                if debug and i < 3:  # 只打印前3个目标
                    print(f"  [损失调试] 目标 {i}: 位置({cx_int},{cy_int}), "
                          f"宽高目标({w_feat:.1f},{h_feat:.1f}), "
                          f"宽高预测({w_pred:.1f},{h_pred:.1f}), 类别{cls_id}")

    if debug:
        print(f"[损失调试] 目标分配: {assigned_objects}/{total_objects} "
              f"({assigned_objects / max(total_objects, 1) * 100:.1f}%)")

    # 新增：宽高回归诊断
    if len(wh_targets_list) > 0 and debug:
        wh_targets = torch.tensor(wh_targets_list)
        wh_preds = torch.tensor(wh_preds_list)

        print(f"[宽高诊断] 目标范围: w[{wh_targets[:, 0].min():.1f}, {wh_targets[:, 0].max():.1f}], "
              f"h[{wh_targets[:, 1].min():.1f}, {wh_targets[:, 1].max():.1f}]")
        print(f"[宽高诊断] 预测范围: w[{wh_preds[:, 0].min():.1f}, {wh_preds[:, 0].max():.1f}], "
              f"h[{wh_preds[:, 1].min():.1f}, {wh_preds[:, 1].max():.1f}]")
        print(f"[宽高诊断] 目标均值: w{wh_targets[:, 0].mean():.1f}, h{wh_targets[:, 1].mean():.1f}")
        print(f"[宽高诊断] 预测均值: w{wh_preds[:, 0].mean():.1f}, h{wh_preds[:, 1].mean():.1f}")

        # 计算平均误差
        wh_errors = torch.abs(wh_preds - wh_targets)
        print(f"[宽高诊断] 平均误差: w{wh_errors[:, 0].mean():.1f}, h{wh_errors[:, 1].mean():.1f}")

        # 检查是否存在极端值
        if wh_targets.max() > 50 or wh_preds.max() > 50:
            print("⚠️ [宽高诊断] 检测到极端宽高值!")
        if wh_errors.mean() > 10:
            print("⚠️ [宽高诊断] 宽高误差过大!")

    # 在计算热力图损失时添加类别权重
    class_weights = torch.tensor([1.5, 0.8, 2.0, 0.7], device=device)  # wall, door, window, column
    if debug:
        # 可视化第一个batch的第一个类别的热力图目标
        import matplotlib.pyplot as plt
        plt.imshow(hm_target[0, 0].cpu().numpy())
        plt.colorbar()
        plt.show()
    # 计算损失
    hm_pred = torch.sigmoid(outputs["heatmap"])
    #hm_loss = focal_loss(hm_pred, hm_target)
    # 分别计算每个类别的损失
    hm_loss_per_class = []
    for cls_id in range(num_classes):
        cls_hm_pred = hm_pred[:, cls_id:cls_id + 1]
        cls_hm_target = hm_target[:, cls_id:cls_id + 1]
        cls_loss = focal_loss(cls_hm_pred, cls_hm_target)
        hm_loss_per_class.append(cls_loss * class_weights[cls_id])

    hm_loss = sum(hm_loss_per_class) / num_classes

    # 只为有目标的位置计算回归损失
    mask_expanded = mask.unsqueeze(1).expand_as(wh_target)
    wh_loss = reg_l1_loss(outputs["wh"], wh_target, mask_expanded)
    off_loss = reg_l1_loss(outputs["offset"], offset_target, mask_expanded)

    total_loss = hm_loss + 0.5 * wh_loss + off_loss

    if debug:
        print(f"[损失调试] 损失值 - hm: {hm_loss:.4f}, wh: {wh_loss:.4f}, off: {off_loss:.4f}")
        print(f"[损失调试] 热力图预测范围: [{hm_pred.min():.3f}, {hm_pred.max():.3f}]")
        print(f"[损失调试] 热力图目标非零数: {(hm_target > 0).sum().item()}")

    loss_dict = {
        "total": total_loss,
        "hm": hm_loss,
        "wh": wh_loss,
        "off": off_loss
    }

    return total_loss, loss_dict


def gaussian_radius(height, width, min_overlap=0.3):
    """计算高斯核半径"""
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = torch.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 - sq1) / (2 * a1)

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = torch.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 - sq2) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = torch.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / (2 * a3)

    return torch.min(torch.min(r1, r2), r3)


def draw_gaussian(heatmap, center, radius, device, k=1):
    """在热力图上绘制高斯核"""
    diameter = 2 * radius + 1
    gaussian = gaussian2d((diameter, diameter), sigma=diameter / 6).to(device)  # 确保高斯核在正确设备上

    x, y = center
    height, width = heatmap.shape

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]

    torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)


def draw_elliptical_gaussian(heatmap, center, sigma_x, sigma_y, device, k=1):
    """
    在热力图上绘制椭圆高斯核。
    """
    # 根据sigma确定绘制区域的大小 (3*sigma 覆盖约99%的区域)
    radius_x = int(3 * sigma_x)
    radius_y = int(3 * sigma_y)

    # 生成网格
    y, x = torch.meshgrid(
        torch.arange(-radius_y, radius_y + 1, dtype=torch.float32, device=device),
        torch.arange(-radius_x, radius_x + 1, dtype=torch.float32, device=device),
        indexing='ij'
    )

    # 计算椭圆高斯: exp(-(x^2/(2*sigma_x^2) + y^2/(2*sigma_y^2)))
    gaussian = torch.exp(-(x * x / (2 * sigma_x * sigma_x) + y * y / (2 * sigma_y * sigma_y)))
    gaussian[gaussian < torch.finfo(gaussian.dtype).eps * gaussian.max()] = 0

    x_c, y_c = center
    height, width = heatmap.shape

    # 确定裁剪边界，防止越界
    left, right = min(x_c, radius_x), min(width - x_c, radius_x + 1)
    top, bottom = min(y_c, radius_y), min(height - y_c, radius_y + 1)

    # 裁剪热力图和高斯核到有效区域
    masked_heatmap = heatmap[y_c - top:y_c + bottom, x_c - left:x_c + right]
    masked_gaussian = gaussian[radius_y - top:radius_y + bottom, radius_x - left:radius_x + right]

    # 使用torch.max进行原地更新
    torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

def gaussian2d(shape, sigma=1):
    """生成2D高斯核"""
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = torch.meshgrid(
        torch.arange(-m, m + 1, dtype=torch.float32),
        torch.arange(-n, n + 1, dtype=torch.float32),
        indexing='ij'
    )
    h = torch.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < torch.finfo(h.dtype).eps * h.max()] = 0
    return h
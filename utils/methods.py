import cv2
import numpy as np
from collections import defaultdict
from PIL import Image
from imagehash import average_hash
from skimage.metrics import structural_similarity as ssim
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import resnet18
from sklearn.metrics.pairwise import cosine_similarity

# Color Histogram
def color_histogram(image_path):
    img = cv2.imread(image_path)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([img_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def group_by_color(image_paths, threshold=0.9):
    hist_groups = defaultdict(list)
    hists = {path: color_histogram(path) for path in image_paths}
    for i, path1 in enumerate(image_paths):
        for path2 in image_paths[i+1:]:
            similarity = cv2.compareHist(hists[path1], hists[path2], cv2.HISTCMP_CORREL)
            if similarity > threshold:
                hist_groups[path1].append(path2)
                hist_groups[path2].append(path1)
    return _build_groups(hist_groups, image_paths)

# pHash
def group_by_phash(image_paths, max_distance=10):
    hash_table = defaultdict(list)
    for path in image_paths:
        img = Image.open(path)
        phash = str(average_hash(img))
        hash_table[phash].append(path)
    return [paths for paths in hash_table.values() if len(paths) > 1]

# SSIM
def group_by_ssim(image_paths, ssim_threshold=0.9):
    similar_pairs = []
    for i, path1 in enumerate(image_paths):
        img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
        for path2 in image_paths[i+1:]:
            img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
            try:
                score = ssim(img1, img2)
                if score > ssim_threshold:
                    similar_pairs.append((path1, path2))
            except Exception:
                continue
    return _group_from_pairs(similar_pairs)

# ORB
def compute_orb_descriptors(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(img, None)
    return descriptors

def group_by_orb(image_paths, match_threshold=50):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    descriptors = {p: compute_orb_descriptors(p) for p in image_paths if compute_orb_descriptors(p) is not None}
    groups = defaultdict(list)
    for i, path1 in enumerate(descriptors):
        for path2 in list(descriptors.keys())[i+1:]:
            try:
                matches = bf.match(descriptors[path1], descriptors[path2])
                if len(matches) > match_threshold:
                    groups[path1].append(path2)
                    groups[path2].append(path1)
            except:
                continue
    return _build_groups(groups, descriptors.keys())

# CNN Feature Matching (optional)
def get_cnn_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.vgg16(pretrained=True).features
    model.eval()
    model.to(device)
    return model, device

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_features(image_path, model, device):
    try:
        img = Image.open(image_path).convert("RGB")
        img_t = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model(img_t)
        return features.flatten().cpu().numpy()
    except Exception:
        return None

def group_by_cnn(image_paths, threshold=0.9):
    model = resnet18(pretrained=True)
    model.eval()
    model = torch.nn.Sequential(*list(model.children())[:-1])  # remove classifier

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(  # Normalization MUST match ImageNet
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )
    ])

    features = []
    for path in image_paths:
        image = Image.open(path).convert('RGB')
        tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            embedding = model(tensor).squeeze().numpy()
        features.append(embedding)

    groups = []
    used = set()

    for i in range(len(image_paths)):
        if i in used:
            continue
        group = [image_paths[i]]
        used.add(i)
        for j in range(i + 1, len(image_paths)):
            if j in used:
                continue
            sim = cosine_similarity(
                features[i].reshape(1, -1), features[j].reshape(1, -1)
            )[0][0]
            if sim >= threshold:
                group.append(image_paths[j])
                used.add(j)
        groups.append(group)

    return groups

# Helpers
def _build_groups(pair_dict, keys):
    visited = set()
    final_groups = []
    for key in keys:
        if key not in visited:
            group = [key] + pair_dict[key]
            if len(group) > 1:
                final_groups.append(group)
            visited.update(group)
    return final_groups

def _group_from_pairs(pairs):
    graph = defaultdict(set)
    for a, b in pairs:
        graph[a].add(b)
        graph[b].add(a)
    visited = set()
    groups = []

    def dfs(node, group):
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                group.append(neighbor)
                dfs(neighbor, group)

    for node in graph:
        if node not in visited:
            group = [node]
            visited.add(node)
            dfs(node, group)
            if len(group) > 1:
                groups.append(group)

    return groups

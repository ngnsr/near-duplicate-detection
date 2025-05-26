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
from typing import List

from PIL import Image
import cv2
import numpy as np
from imagehash import average_hash
from skimage.metrics import structural_similarity as ssim
import torch
import torchvision.models as models
import torchvision.transforms as transforms

def compare_by_color(img_path1, img_path2, threshold=0.9):
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)
    img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    hist1 = cv2.calcHist([img1_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist2 = cv2.calcHist([img2_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return similarity

def compare_by_ahash(img_path1, img_path2, max_distance=10):
    img1 = Image.open(img_path1)
    img2 = Image.open(img_path2)
    hash1 = average_hash(img1)
    hash2 = average_hash(img2)
    distance = hash1 - hash2
    return distance

def compare_by_ssim(img_path1, img_path2, ssim_threshold=0.9):
    img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)
    img1 = cv2.resize(img1, (224, 224))
    img2 = cv2.resize(img2, (224, 224))
    similarity, _ = ssim(img1, img2, full=True)
    return similarity

def compare_by_orb(img_path1, img_path2, match_threshold=50):
    img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)
    img1 = cv2.resize(img1, (224, 224))
    img2 = cv2.resize(img2, (224, 224))
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    if des1 is None or des2 is None:
        return 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    return len(good_matches)

def compare_by_cnn(img_path1, img_path2, threshold=0.9):
    model = models.resnet18(pretrained=True)
    model.eval()
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img1 = Image.open(img_path1).convert("RGB")
    img2 = Image.open(img_path2).convert("RGB")
    img1_tensor = transform(img1).unsqueeze(0)
    img2_tensor = transform(img2).unsqueeze(0)
    with torch.no_grad():
        feat1 = model(img1_tensor).flatten()
        feat2 = model(img2_tensor).flatten()
    similarity = torch.cosine_similarity(feat1, feat2, dim=0).item()
    return similarity

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
def hamming_distance(hash1, hash2):
    return bin(int(str(hash1), 16) ^ int(str(hash2), 16)).count('1')

def group_by_ahash(image_paths, max_distance=10):
    hashes = []
    for path in image_paths:
        img = Image.open(path)
        ahash = average_hash(img)
        hashes.append((path, ahash))

    visited = set()
    groups = []

    for i in range(len(hashes)):
        if hashes[i][0] in visited:
            continue
        group = [hashes[i][0]]
        visited.add(hashes[i][0])
        for j in range(i + 1, len(hashes)):
            if hashes[j][0] in visited:
                continue
            dist = hamming_distance(hashes[i][1], hashes[j][1])
            if dist <= max_distance:
                group.append(hashes[j][0])
                visited.add(hashes[j][0])
        if len(group) > 1:
            groups.append(group)

    return groups

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

def compute_orb_descriptors(image_path: str, max_size: int = 300, nfeatures: int = 300) -> tuple:
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        h, w = img.shape
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        
        img = cv2.equalizeHist(img)
        orb = cv2.ORB_create(nfeatures=nfeatures, scaleFactor=1.2, nlevels=8)
        keypoints, descriptors = orb.detectAndCompute(img, None)
        
        if descriptors is None or len(keypoints) < 10:
            raise ValueError(f"Insufficient keypoints ({len(keypoints)}) for {image_path}")
        
        return keypoints, descriptors
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None

def group_by_orb(image_paths: List[str], match_threshold: float = 0.6, min_matches: int = 40, max_distance: int = 50) -> List[List[str]]:
    descriptors_dict = {}
    for path in image_paths:
        keypoints, descriptors = compute_orb_descriptors(path)
        if descriptors is not None:
            descriptors_dict[path] = descriptors
    
    if not descriptors_dict:
        return []

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    similar_pairs = []
    
    for i, path1 in enumerate(descriptors_dict):
        desc1 = descriptors_dict[path1]
        for path2 in list(descriptors_dict.keys())[i+1:]:
            desc2 = descriptors_dict[path2]
            try:
                matches = bf.knnMatch(desc1, desc2, k=2)
                good_matches = [
                    m for m, n in matches 
                    if m.distance < match_threshold * n.distance and m.distance < max_distance
                ]
                if len(good_matches) >= min_matches:
                    similar_pairs.append((path1, path2))
                    print(f"Match {path1} and {path2}: {len(good_matches)} good matches")
            except Exception as e:
                print(f"Error matching {path1} and {path2}: {e}")
                continue
    
    graph = defaultdict(set)
    for a, b in similar_pairs:
        graph[a].add(b)
        graph[b].add(a)
    
    visited = set()
    groups = []
    
    def dfs(node: str, group: List[str]):
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                group.append(neighbor)
                dfs(neighbor, group)
    
    for node in descriptors_dict:
        if node not in visited:
            group = [node]
            visited.add(node)
            dfs(node, group)
            # Filter groups with weak overall similarity
            if len(group) > 1:
                avg_match_strength = sum(
                    len([m for m, n in bf.knnMatch(descriptors_dict[node], descriptors_dict[neighbor], k=2)
                         if m.distance < match_threshold * n.distance and m.distance < max_distance])
                    for neighbor in group if node != neighbor
                ) / max(1, len(group) - 1)
                if avg_match_strength >= min_matches * 0.5:  # Require half the min_matches as average
                    groups.append(group)
    
    return groups

# CNN Feature Matching
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

    return [g for g in groups if len(g) > 1]

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

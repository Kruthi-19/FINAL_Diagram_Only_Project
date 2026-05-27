import cv2, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
GRID_SIZE=40
RESULT_DIR="static/results"

def create_spatial_map(image_path, detections):
    img=cv2.imread(image_path); h,w=img.shape[:2]
    grid=np.zeros((GRID_SIZE,GRID_SIZE), dtype=int)
    zone_map=[]
    obstacle_labels={"stage","pillar","helpdesk"}
    for det in detections:
        label=det["label"]; x1,y1,x2,y2=det["bbox"]
        gx1=int((x1/w)*GRID_SIZE); gy1=int((y1/h)*GRID_SIZE)
        gx2=int((x2/w)*GRID_SIZE); gy2=int((y2/h)*GRID_SIZE)
        gx1,gy1=max(0,gx1),max(0,gy1); gx2,gy2=min(GRID_SIZE-1,gx2),min(GRID_SIZE-1,gy2)
        if label in obstacle_labels:
            grid[gy1:gy2+1,gx1:gx2+1]=1
        zone_map.append({"label":label,"grid_bbox":[gx1,gy1,gx2,gy2],"center":((gy1+gy2)//2,(gx1+gx2)//2)})
    out=save_grid(image_path,grid)
    return grid, zone_map, out

def save_grid(image_path,grid):
    name=os.path.splitext(os.path.basename(image_path))[0]
    out=os.path.join(RESULT_DIR,f"{name}_02_spatial_map.png")
    plt.figure(figsize=(6,5)); plt.imshow(grid,cmap="gray_r")
    plt.title("Spatial Mapping: White = Walkable, Black = Blocked")
    plt.axis("off"); plt.savefig(out,bbox_inches="tight"); plt.close()
    return out

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
RESULT_DIR="static/results"

def draw_candidate_layouts(image_path,grid,zone_map,layouts,best):
    name=os.path.splitext(os.path.basename(image_path))[0]
    out=os.path.join(RESULT_DIR,f"{name}_05_layout_candidates.png")
    fig,axes=plt.subplots(1,len(layouts),figsize=(18,4))
    for ax,layout in zip(axes,layouts):
        draw_layout(ax,grid,zone_map,layout,layout["name"]==best["name"])
        ax.set_title(f"{layout['name']}\nScore: {layout['fitness']}")
    plt.tight_layout(); plt.savefig(out,bbox_inches="tight"); plt.close(); return out

def draw_best_layout(image_path,grid,zone_map,best):
    name=os.path.splitext(os.path.basename(image_path))[0]
    out=os.path.join(RESULT_DIR,f"{name}_06_optimized_layout.png")
    fig,ax=plt.subplots(figsize=(7,6)); draw_layout(ax,grid,zone_map,best,True)
    ax.set_title(f"Final Optimized Layout: {best['name']} | Score: {best['fitness']}/100")
    plt.savefig(out,bbox_inches="tight"); plt.close(); return out

def draw_layout(ax,grid,zone_map,layout,is_best=False):
    ax.set_xlim(0,grid.shape[1]); ax.set_ylim(grid.shape[0],0)
    ax.add_patch(plt.Rectangle((0,0),grid.shape[1]-1,grid.shape[0]-1,fill=False,linewidth=2))
    colors={"stage":"#8e44ad","exit":"#27ae60","entry":"#3498db","pillar":"#555555","helpdesk":"#2980b9"}
    for z in zone_map:
        x1,y1,x2,y2=z["grid_bbox"]; label=z["label"]
        ax.add_patch(plt.Rectangle((x1,y1),max(1,x2-x1),max(1,y2-y1),color=colors.get(label,"#ccc"),alpha=.85))
        ax.text((x1+x2)/2,(y1+y2)/2,label.upper(),color="white",ha="center",va="center",fontsize=6)
    for i,(y,x) in enumerate(layout["stalls"],1):
        ax.add_patch(plt.Rectangle((x-1.3,y-1),2.6,2,color="#f39c12",alpha=.95))
        ax.text(x,y,f"S{i}",ha="center",va="center",fontsize=6)
    entries=[z["center"] for z in zone_map if z["label"]=="entry"]; exits=[z["center"] for z in zone_map if z["label"]=="exit"]
    if entries and exits:
        ey,ex=entries[0]
        for gy,gx in exits:
            ax.plot([ex,grid.shape[1]/2,gx],[ey,grid.shape[0]/2,gy],"--",color="green",linewidth=1.5)
    if is_best: ax.text(2,grid.shape[0]-2,"BEST",color="green",fontsize=10,weight="bold")
    ax.axis("off")

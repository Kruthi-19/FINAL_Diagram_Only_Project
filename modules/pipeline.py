from modules.detector import detect_venue_objects
from modules.spatial import create_spatial_map
from modules.risk_path import generate_kde_heatmap, generate_multiple_paths
from modules.layouts import optimize_layouts
from modules.visualizer import draw_candidate_layouts, draw_best_layout

def run_project_pipeline(image_path,event_data):
    detections,detected_img=detect_venue_objects(image_path,event_data)
    grid,zone_map,spatial_img=create_spatial_map(image_path,detections)
    density,heatmap_img=generate_kde_heatmap(image_path,grid,zone_map,event_data)
    path_options,best_path,path_img=generate_multiple_paths(image_path,grid,density,zone_map)
    layouts,best_layout=optimize_layouts(grid,zone_map,density,event_data)
    candidates_img=draw_candidate_layouts(image_path,grid,zone_map,layouts,best_layout)
    optimized_img=draw_best_layout(image_path,grid,zone_map,best_layout)
    return {
        "detected_img":detected_img,"spatial_img":spatial_img,"heatmap_img":heatmap_img,
        "path_img":path_img,"candidates_img":candidates_img,"optimized_img":optimized_img,
        "detections":detections,"path_options":path_options,"best_path":best_path,
        "layouts":layouts,"best_layout":best_layout,"final_score":int(best_layout["fitness"]),
        "event_data":event_data
    }

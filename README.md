# AI-Driven Venue Layout Optimizer

Run:
python -m venv venv
venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py

Open: http://127.0.0.1:5000

This Phase-1 prototype follows your PPT workflow:
User input -> YOLO-ready detection -> Spatial Mapping -> KDE -> A* -> SLP + GA-ACO -> Optimized Layout.
Current detection uses a YOLO-ready fallback because final YOLO training requires a custom annotated event-hall dataset.


## UI Update
The result page is beautified so each algorithm output appears in a separate full-width presentation section.


## Separate Output Dashboard
Each algorithm output now has an 'Open Separate Dashboard' button. It opens a clean individual dashboard page for that stage.

from flask import Flask, render_template, request
import os, uuid
from modules.pipeline import run_project_pipeline

app = Flask(__name__)
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/results", exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    result, error = None, None
    if request.method == "POST":
        try:
            file = request.files.get("venue_image")
            if not file or file.filename == "":
                return render_template("index.html", result=None, error="Please upload a hall image.")

            filename = f"{uuid.uuid4().hex}_{file.filename}"
            image_path = os.path.join("static/uploads", filename)
            file.save(image_path)

            event_data = {
                "stalls": int(request.form.get("stalls", 8)),
                "crowd": int(request.form.get("crowd", 300)),
                "stage_size": request.form.get("stage_size", "Medium"),
                "exit_positions": request.form.get("exit_positions", "Left and Right")
            }

            result = run_project_pipeline(image_path, event_data)

        except Exception as e:
            error = str(e)

    return render_template("index.html", result=result, error=error)


@app.route("/output")
def output_view():
    """
    Separate dashboard page for viewing one algorithm output at a time.
    Values are passed from the main dashboard as query parameters.
    """
    title = request.args.get("title", "Algorithm Output")
    subtitle = request.args.get("subtitle", "")
    image = request.args.get("image", "")
    explanation = request.args.get("explanation", "")
    step = request.args.get("step", "")
    return render_template(
        "output.html",
        title=title,
        subtitle=subtitle,
        image=image,
        explanation=explanation,
        step=step
    )


if __name__ == "__main__":
    app.run(debug=True)

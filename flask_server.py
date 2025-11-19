from flask import Flask, Response, render_template_string
import threading
import time
import io
import cv2
import main_app

app = Flask(__name__)

HTML = """<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>Inference Stream</title>
    <style>
      body { font-family: Arial, Helvetica, sans-serif; background:#222; color:#eee; display:flex; flex-direction:column; align-items:center; }
      .container { margin-top:20px; }
      img { border: 4px solid #444; box-shadow: 0 4px 12px rgba(0,0,0,0.6); }
      .info { margin-top:8px; }
    </style>
  </head>
  <body>
    <div class="container">
      <h2>Inference Stream</h2>
      <img id="frame" src="/stream" style="width:%(w)spx; height:auto;" />
      <div class="info">Stream resolution (target): %(w)s x %(h)s — page scaled to width</div>
    </div>
  </body>
</html>
""" % {"w": main_app.TARGET_WIDTH, "h": main_app.TARGET_HEIGHT}


def gen_frames():
    """Generator that yields MJPEG frames from the latest frame in `main_app`."""
    while True:
        frame = main_app.get_latest_frame()
        if frame is None:
            time.sleep(0.05)
            continue
        try:
            ret, jpg = cv2.imencode('.jpg', frame)
            if not ret:
                time.sleep(0.05)
                continue
            data = jpg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + data + b'\r\n')
        except GeneratorExit:
            break
        except Exception:
            time.sleep(0.05)
            continue


@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/stream')
def stream():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def _run_flask():
    # Do not use reloader (it spawns a child process which complicates OpenCV windows)
    app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)


if __name__ == '__main__':
    # Start Flask in a background daemon thread, then run the main app in the main thread
    t = threading.Thread(target=_run_flask, daemon=True)
    t.start()
    print('Flask server started on http://127.0.0.1:5000')
    print('Now starting main_app (will open OpenCV window for inference).')
    # Run the main app in the main thread to ensure OpenCV windows behave correctly on Windows
    main_app.main()

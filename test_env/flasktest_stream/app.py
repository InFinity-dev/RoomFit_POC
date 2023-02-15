from flask import Flask, render_template_string
import time

app = Flask(__name__)

@app.route('/')
def index():
    return render_template_string('''<p>This is the current value: <span id="latest_value"></span></p>
<script>

    var latest = document.getElementById('latest_value');

    var xhr = new XMLHttpRequest();
    xhr.open('GET', '{{ url_for('stream') }}');

    xhr.onreadystatechange = function() {
        var all_lines = xhr.responseText.split('\\n');
        last_line = all_lines.length - 2
        latest.textContent = all_lines[last_line]

        if (xhr.readyState == XMLHttpRequest.DONE) {
            /*alert("The End of Stream");*/
            latest.textContent = "The End of Stream"
        }
    }

    xhr.send();

</script>''')

@app.route('/stream_time')
def stream():
    def generate():
        while True:
            current_time = time.strftime("%H:%M:%S\n")
            print(current_time)
            yield current_time
            time.sleep(1)

    return app.response_class(generate(), mimetype='text/plain')

app.run()
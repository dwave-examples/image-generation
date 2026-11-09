# Copyright 2026 D-Wave
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging

from dash import Dash
from flask import send_from_directory

from demo_configs import APP_TITLE, ROOT
from demo_interface import create_interface
import demo_callbacks  # noqa: F401 - registers Dash callbacks

app = Dash(__name__, title=APP_TITLE)
app.layout = create_interface()
server = app.server


@server.route("/static/<path:filename>")
def static_file(filename: str):
    return send_from_directory(ROOT / "static", filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pretrained FFHQ image-generation demo."
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    app.run(debug=args.debug, dev_tools_hot_reload=False, use_reloader=False)

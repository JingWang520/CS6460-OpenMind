import re
import subprocess
import random
import string
from openai import OpenAI
import os
from pathlib import Path
from playwright.sync_api import sync_playwright, Browser, Page


class MindmapGenerator:
    TMP_DIR = "tmp_dir"
    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(self, api_key: str ="sk-JWU9jpEGAv2kHo3YBNB1TPIuhVjT4Nnf60GT55n7iz5GY3g7", base_url: str = "https://www.dmxapi.com/v1"):
        # Initialize OpenAI client
        try:
            self.client = OpenAI(base_url=base_url, api_key=api_key)
            print("OpenAI client initialized successfully.")
        except Exception as e:
            print(f"Error initializing OpenAI client: {e}")
            self.client = None

        # Delay Playwright browser initialization to avoid unnecessary overhead
        self._playwright = None
        self._browser: Browser = None

    def _init_browser(self):
        if self._playwright is None:
            self._playwright = sync_playwright().start()
        if self._browser is None:
            self._browser = self._playwright.chromium.launch()

    def close_browser(self):
        """Close browser and Playwright, release resources"""
        if self._browser:
            self._browser.close()
            self._browser = None
        if self._playwright:
            self._playwright.stop()
            self._playwright = None

    @staticmethod
    def _random_suffix(length=6):
        return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

    def generate_response(self, prompt, model=None, max_tokens=500):
        """Wrapper for OpenAI generation interface"""
        if model is None:
            model = self.DEFAULT_MODEL
        if not self.client:
            return "Error: OpenAI client not initialized."
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system",
                     "content": "You are a helpful assistant specialized in creating structured Markdown content."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"API Error: {str(e)}"

    def generate_mindmap_md(self, topic, model=None):
        """Generate mind map structure in Markdown format"""
        prompt = f"""Generate a hierarchical mind map structure in MARKDOWN LIST FORMAT for the topic: "{topic}".
Use proper Markdown list indentation with spaces (2 spaces per level).
The output should ONLY contain valid Markdown lists, no additional text.

Example output format:
- Main Topic 1
  - Sub Topic 1
    - Detail 1
    - Detail 2
  - Sub Topic 2
- Main Topic 2
  - Sub Topic 3
    - Detail 3

Now generate for topic:
{topic}
"""
        response = self.generate_response(prompt, model=model)
        # Validity check
        if response.startswith("Error"):
            return response
        if not any(c in response for c in ['*', '-']):
            return f"Error: Invalid Markdown format in response:\n{response}"

        return response

    def save_markdown(self, content, filename_prefix="mindmap"):
        """Save Markdown file to temporary directory with random suffix in filename"""
        Path(self.TMP_DIR).mkdir(exist_ok=True)
        random_part = self._random_suffix()
        filename = f"{filename_prefix}_{random_part}.md"
        filepath = os.path.join(self.TMP_DIR, filename)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return filepath
        except Exception as e:
            return f"Error saving Markdown: {str(e)}"

    def convert_md_to_png(self, md_path, output_png=None, viewport_width=1200, viewport_height=800, full_page=True):
        """
        Convert Markdown file to PNG using markmap + Playwright
        """
        if not os.path.exists(md_path):
            return f"Error: File not found: {md_path}"

        # 1. Convert Markdown to HTML (using markmap)
        html_path = Path(md_path).with_suffix('.html')
        try:
            cmd_str = f"markmap {md_path} --output {html_path} --no-open"

            result = subprocess.run(
                cmd_str,
                check=True,
                shell=True,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                return f"markmap conversion failed: {result.stderr}"

        except subprocess.CalledProcessError as e:
            return f"markmap error: {str(e)}"
        except Exception as e:
            return f"Error during markmap conversion: {str(e)}"

        # 2. Use Playwright to convert HTML to PNG
        if output_png is None:
            random_part = self._random_suffix()
            output_png = Path(self.TMP_DIR) / f"{Path(md_path).stem}_{random_part}.png"
        else:
            output_png = Path(output_png)

        try:
            self._init_browser()
            page: Page = self._browser.new_page()
            page.set_viewport_size({"width": viewport_width, "height": viewport_height})

            file_uri = html_path.resolve().as_uri()
            page.goto(file_uri, wait_until='networkidle')

            # Remove toolbar
            page.evaluate('''() => {
                const toolbar = document.querySelector('div.mm-toolbar');
                if (toolbar) toolbar.remove();
            }''')

            # Wait for map rendering to complete
            page.wait_for_selector("#mindmap", timeout=5000)

            screenshot_options = {
                "path": str(output_png),
                "full_page": full_page,
                "type": "png"
            }

            if not full_page:
                screenshot_options["clip"] = {
                    "x": 0, "y": 0,
                    "width": viewport_width,
                    "height": viewport_height
                }

            page.screenshot(**screenshot_options)
            page.close()

            return str(output_png.resolve())

        except Exception as e:
            return f"Playwright conversion error: {str(e)}"

    def generate_full_mindmap(self, topic, filename_prefix="mindmap"):
        """
        Complete in one step: generate Markdown, save file, convert to PNG
        """
        md_content = self.generate_mindmap_md(topic)
        if md_content.startswith("Error"):
            return md_content, None

        md_path = self.save_markdown(md_content, filename_prefix=filename_prefix)
        if isinstance(md_path, str) and md_path.startswith("Error"):
            return md_path, None

        png_path = self.convert_md_to_png(md_path)
        return md_path, png_path


if __name__ == "__main__":
    # Example usage
    api_key = "sk-JWU9jpEGAv2kHo3YBNB1TPIuhVjT4Nnf60GT55n7iz5GY3g7"

    mg = MindmapGenerator(api_key=api_key)

    topic = "Quantum Computing Fundamentals"

    # Example of single step calls
    md_content = mg.generate_mindmap_md(topic)
    if not md_content.startswith("Error"):
        md_path = mg.save_markdown(md_content)
        png_path = mg.convert_md_to_png(md_path)
        print(f"Single steps results:\nMarkdown: {md_path}\nPNG: {png_path}")
    else:
        print(md_content)

    # Example of full generation call
    md_path_full, png_path_full = mg.generate_full_mindmap(topic)
    print(f"Full generation results:\nMarkdown: {md_path_full}\nPNG: {png_path_full}")

    # Close browser to release resources
    mg.close_browser()

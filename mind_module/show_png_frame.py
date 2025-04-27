import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import platform

class ZoomPanImageViewer(tk.Frame):
    """
    一个支持缩放和平移图像的Tkinter Frame组件。
    """
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)

        # --- 组件设置 ---
        # 将 bg="gray" 修改为 bg="white"
        self.canvas = tk.Canvas(self, bg="white", highlightthickness=0) # <--- 修改这里
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # --- 图片相关属性 ---
        self.original_image = None
        self.tk_image = None
        self.image_id = None
        self.zoom_level = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 5.0

        # --- 拖动相关属性 ---
        self._drag_start_x = 0
        self._drag_start_y = 0
        self._is_dragging = False

        # --- 事件绑定 ---
        self.canvas.bind("<ButtonPress-1>", self._on_button_press)
        self.canvas.bind("<B1-Motion>", self._on_button_motion)
        self.canvas.bind("<ButtonRelease-1>", self._on_button_release)

        if platform.system() == "Windows" or platform.system() == "Darwin":
             self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        else:
             self.canvas.bind("<Button-4>", self._on_mouse_wheel)
             self.canvas.bind("<Button-5>", self._on_mouse_wheel)

        self.canvas.bind("<Configure>", self._on_configure)
        self.canvas.focus_set()

    # ... (其他方法 load_image, _clear_image, _display_image, 等保持不变) ...
    def load_image(self, filepath):
        """加载指定路径的图片"""
        try:
            self.original_image = Image.open(filepath)
            print(f"图片加载成功: {filepath}, 尺寸: {self.original_image.size}")
            self.zoom_level = 1.0 # 重置缩放级别
            self._display_image() # 显示图片
        except FileNotFoundError:
            print(f"错误: 文件未找到 {filepath}")
            self._clear_image()
        except Exception as e:
            print(f"加载图片时出错: {e}")
            self._clear_image()

    def _clear_image(self):
        """清除当前显示的图片"""
        if self.image_id:
            self.canvas.delete(self.image_id)
        self.original_image = None
        self.tk_image = None
        self.image_id = None
        self.zoom_level = 1.0

    def _display_image(self):
        """根据当前缩放级别显示图片"""
        if not self.original_image:
            return

        if self.image_id:
            self.canvas.delete(self.image_id)

        width = max(1, int(self.original_image.width * self.zoom_level))
        height = max(1, int(self.original_image.height * self.zoom_level))

        try:
            resized_img = self.original_image.resize((width, height), Image.Resampling.LANCZOS)
            self.tk_image = ImageTk.PhotoImage(resized_img)
            self.image_id = self.canvas.create_image(0, 0, anchor=tk.CENTER, image=self.tk_image)
            self.canvas.image = self.tk_image # Keep reference
            self._center_image()
        except Exception as e:
            print(f"调整大小或显示图片时出错: {e}")
            self.image_id = None
            self.tk_image = None

    def _center_image(self):
        """将图片置于Canvas中心"""
        if not self.image_id:
            return
        self.canvas.update_idletasks()
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width > 1 and canvas_height > 1:
            self.canvas.coords(self.image_id, canvas_width / 2, canvas_height / 2)

    def _on_configure(self, event):
        """当Canvas大小改变时，重新居中图片"""
        self._center_image()

    def _on_button_press(self, event):
        """处理鼠标左键按下事件 (开始拖动)"""
        if not self.image_id: return
        self._drag_start_x = event.x
        self._drag_start_y = event.y
        self._is_dragging = True
        self.canvas.config(cursor="fleur")

    def _on_button_motion(self, event):
        """处理鼠标拖动事件"""
        if not self.image_id or not self._is_dragging: return
        dx = event.x - self._drag_start_x
        dy = event.y - self._drag_start_y
        self.canvas.move(self.image_id, dx, dy)
        self._drag_start_x = event.x
        self._drag_start_y = event.y

    def _on_button_release(self, event):
        """处理鼠标左键释放事件 (结束拖动)"""
        if not self.image_id: return
        self._is_dragging = False
        self.canvas.config(cursor="")

    def _on_mouse_wheel(self, event):
        """处理鼠标滚轮事件 (缩放)"""
        if not self.image_id: return

        factor = 0.0
        if platform.system() == "Windows" or platform.system() == "Darwin":
            factor = 1.1 if event.delta > 0 else 0.9
        else: # Linux
            if event.num == 4: factor = 1.1
            elif event.num == 5: factor = 0.9

        if factor == 0.0: return

        new_zoom_level = self.zoom_level * factor
        new_zoom_level = max(self.min_zoom, min(self.max_zoom, new_zoom_level))

        if new_zoom_level == self.zoom_level: return

        actual_factor = new_zoom_level / self.zoom_level
        self.zoom_level = new_zoom_level

        mouse_x, mouse_y = event.x, event.y
        img_x, img_y = self.canvas.coords(self.image_id)

        offset_x = mouse_x - img_x
        offset_y = mouse_y - img_y
        new_offset_x = offset_x * actual_factor
        new_offset_y = offset_y * actual_factor
        new_img_x = mouse_x - new_offset_x
        new_img_y = mouse_y - new_offset_y

        new_width = max(1, int(self.original_image.width * self.zoom_level))
        new_height = max(1, int(self.original_image.height * self.zoom_level))

        try:
            resized_img = self.original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            self.tk_image = ImageTk.PhotoImage(resized_img)
        except Exception as e:
            print(f"缩放时调整图片大小出错: {e}")
            self.zoom_level /= actual_factor # Revert zoom level
            return

        self.canvas.itemconfig(self.image_id, image=self.tk_image)
        self.canvas.coords(self.image_id, new_img_x, new_img_y)
        self.canvas.image = self.tk_image # Keep reference


# --- 示例用法 (也移除了 Frame 的背景色设置，使其更清晰) ---
if __name__ == "__main__":
    root = tk.Tk()
    root.title("图片查看器 (缩放/拖动 - 白色背景)")
    root.geometry("800x600")

    # 创建图片查看器实例 (不再设置Frame的背景色，让Canvas的白色背景更明显)
    image_viewer = ZoomPanImageViewer(root) # <--- 移除了 bg="dark slate gray"
    image_viewer.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def open_image_file():
        filepath = filedialog.askopenfilename(
            title="选择PNG图片",
            filetypes=[("PNG 文件", "*.png"), ("JPEG 文件", "*.jpg;*.jpeg"), ("所有文件", "*.*")]
        )
        if filepath:
            image_viewer.load_image(filepath)

    button_frame = tk.Frame(root)
    button_frame.pack(pady=5)
    open_button = ttk.Button(button_frame, text="打开图片", command=open_image_file)
    open_button.pack()

    default_image_path = "test_image.png" # <--- 修改为你想要测试的图片路径
    try:
        image_viewer.load_image(default_image_path)
    except Exception as e:
         print(f"无法加载默认图片 '{default_image_path}': {e}")
         print("请点击 '打开图片' 按钮选择一个文件。")

    root.mainloop()


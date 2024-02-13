import tkinter as tk
import tkinter as tk
from tkinter import ttk
import imageio
from PIL import Image, ImageTk

class LoadingScreen:
    def __init__(self, master):
        self.master = master
        self.loading_gif = imageio.mimread("loading.gif")
        self.loading_gif = [self.resize_image(Image.fromarray(frame), width=30, height=30) for frame in self.loading_gif]
        self.idx = 0

        self.load_button = ttk.Button(master, text="Load", command=self.simulate_loading)
        self.load_button.pack(pady=20)

    def resize_image(self, image, width, height):
        return image.resize((width, height), Image.ANTIALIAS)

    def simulate_loading(self):
        # Replace button with loading animation
        self.load_button.pack_forget()
        self.loading_label = tk.Label(self.master)
        self.loading_label.pack(padx=20, pady=20)
        self.animate()

        # Simulate a task that takes 3 seconds
        self.master.after(3000, self.close_loading)

    def animate(self):
        self.idx += 1
        if self.idx >= len(self.loading_gif):
            self.idx = 0
        img = ImageTk.PhotoImage(self.loading_gif[self.idx])
        self.loading_label.config(image=img)
        self.loading_label.image = img  # Keep a reference to prevent garbage collection
        self.master.after(100, self.animate)

    def close_loading(self):
        self.loading_label.pack_forget()
        self.load_button.pack(pady=20)

# Create the main Tkinter window
root = tk.Tk()
root.title("Main Window")

# Create an instance of the LoadingScreen class
loading_screen = LoadingScreen(root)

# Run the Tkinter event loop
root.mainloop()
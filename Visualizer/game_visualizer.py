import tkinter as tk

### Imports ###

class GameVisualizer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("DRG Barrel AI v1.0")
        self.root.configure(bg='black')

        self.score_label = tk.Label(self.root, text="Score: 0", fg="white", bg="black")
        self.score_label.pack()

        self.kicks_progress = tk.Canvas(self.root, width=200, height=20, bg="grey")
        self.kicks_progress.pack()
        self.update_progress_bar(0)

        self.action_label = tk.Label(self.root, text="Last Action: None", fg="white", bg="black")
        self.action_label.pack()

    def update_progress_bar(self, kicks):
        self.kicks_progress.delete("progress")
        progress = (kicks / 100.0) * 200
        self.kicks_progress.create_rectangle(0, 0, progress, 20, fill="green", tags="progress")

    def update_info(self, score, kicks, action):
        self.score_label.config(text=f"Score: {score}")
        self.update_progress_bar(kicks)
        self.action_label.config(text=f"Last Action: {action}")

    def run(self):
        self.root.mainloop()

    def close(self):
        self.root.after(0, self.root.destroy)
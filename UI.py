import tkinter as tk
from tkinter import filedialog as fd
import os

root = tk.Tk()

def main():
    root.title("Starter UI Window")
    root.geometry("1000x800")  # Width x Height

    label = tk.Label(root,
                     text="Welcome to Shield-Me, a program to generate IO shield models",
                     fg="blue",
                     font=("Arial", 20))
    label.pack(pady=5)

    frame = tk.Frame(root)
    root.button = tk.Button(root, text="upload file", command = upload_file)
    root.button.pack(pady=(5,0))

    root.mainloop()

def upload_file():

    filepath = fd.askopenfilename()
    if filepath:
        try: root.fileLabel.destroy()
        except: 
            pass
        fileName = os.path.basename(filepath)
        root.fileLabel = tk.Label(root, text=fileName, anchor="w", width=100)
        root.fileLabel.pack(side='left', padx=(550, 0))
    return

if __name__ == "__main__":
    main()

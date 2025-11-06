import tkinter as tk
from tkinter import messagebox, filedialog
import customtkinter as ctk
from volume_calculator_backend import *
from functools import lru_cache
import numpy as np
from PIL import Image, ImageTk
from collections import deque

def start():
    background_color = "#c5d2f7"  

    # Initialize customtkinter appearance
    ctk.set_appearance_mode("system")  # Modes: "light", "dark", "system"
    ctk.set_default_color_theme("blue")  # Themes: "blue", "dark-blue", "green"

    window = ctk.CTk()
    window.title("MRI Volume Calculator")
    window.geometry("800x400")
    window.attributes("-fullscreen", True)
    def exit_fullscreen():
        window.attributes("-fullscreen", False)
        window.geometry("800x400")
    window.bind("<Escape>", lambda x: exit_fullscreen())
    window.configure(fg_color=background_color)  # soft background

    # Example stub functions for commands (replace with your actual functions)
    def choose_directory():
        path = filedialog.askdirectory()
        if path:
            directory_chosen.set(path)


    def choose_download_file():
        path = filedialog.askopenfilename(
            title="Select Excel File",
            filetypes=[("Excel Files", "*.xlsx *.xls"), ("All Files", "*.*")]
        )
        if path:
            volume_download_file.set(path)

    def download_volumes():
        try:
            write_volume_to_file(directory_chosen.get(), volume_download_file.get(), int(image_height.get()), int(image_width.get()), int(image_depth.get()))
        except ValueError:
            messagebox.showerror("Error", "Error has occured: Perhaps you didn't enter correct measurement values")
        except:
            messagebox.showerror("Error", "Error has occured: Perhaps you didn't specify an excel file")

    new_frame = False
        

    class ImageDisplayer:
        def __init__(self, parent_window, background_color):
            self.window = parent_window
            self.background_color = background_color
            self.images = []
            self.current_img_index = -1
            self.mask_frame = None
            self.images_tk = []
            self.mask_directory_chosen = tk.StringVar()
            self._init_ui()

        def _init_ui(self):
            # Main container
            self.window.images_frame = ctk.CTkFrame(self.window, fg_color=self.background_color, border_width=0)
            self.window.images_frame.pack(pady=10, fill='both', padx=20)

            # Buttons frame
            buttons_frame2 = ctk.CTkFrame(self.window.images_frame, fg_color=self.background_color, border_width=0)
            buttons_frame2.pack(pady=20, fill='x', padx=20)

            # Buttons
            ctk.CTkButton(buttons_frame2, text="Prev", command=self.display_prev, corner_radius=12,
                        fg_color='#f44336', hover_color="#B01414", text_color='white',
                        font=ctk.CTkFont(size=16, weight="bold"), width=150).pack(side='left', pady=5, padx=(0, 20))

            ctk.CTkButton(buttons_frame2, text="Next", command=self.display_next, corner_radius=12,
                        fg_color='#4CAF50', hover_color="#179D17", text_color='white',
                        font=ctk.CTkFont(size=16, weight="bold"), width=150).pack(side='left', pady=5, padx=(0, 20))

            ctk.CTkButton(buttons_frame2, text="Choose download location", command=self.choose_mask_directory,
                        corner_radius=12, fg_color='#4CAF50', hover_color="#179D17", text_color='white',
                        font=ctk.CTkFont(size=16, weight="bold"), width=150).pack(side='left', pady=5, padx=(0, 20))

            ctk.CTkLabel(buttons_frame2, textvariable=self.mask_directory_chosen, fg_color="white",
                        text_color="#555555", anchor='w', corner_radius=6, width=400).pack(side='left', pady=5, padx=(0, 20))

            ctk.CTkButton(buttons_frame2, text="Download masks", command=self.download_masks,
                        corner_radius=12, fg_color='#4CAF50', hover_color="#179D17", text_color='white',
                        font=ctk.CTkFont(size=16, weight="bold"), width=180).pack(side='left', pady=5, padx=(0, 20))

            # Load image paths
            image_files = os.listdir(directory_chosen.get())
            self.images = [os.path.join(directory_chosen.get(), file) for file in image_files if file.endswith('.nii')]

            self.display_next(destroy=False)

        def choose_mask_directory(self):
            path = filedialog.askdirectory()
            if path:
                self.mask_directory_chosen.set(path)

        def download_masks(self):
            # Placeholder
            pass

        def display(self, images_tk, destroy=True):
            if destroy and self.mask_frame is not None:
                self.mask_frame.destroy()
                

            self.mask_frame = ctk.CTkFrame(self.window.images_frame, fg_color=self.background_color, border_width=0)
            self.mask_frame.pack(pady=20, fill='x', padx=20)

            labels_text = ['MRI first slice', 'mask first slice', 'MRI middle slice', 'mask middle slice']
            for i in range(4):
                row, col = i // 2, i % 2

                label = ctk.CTkLabel(self.mask_frame, text=labels_text[i],
                                    font=ctk.CTkFont(size=16, weight="bold"))
                label.grid(row=row * 2, column=col, padx=20, pady=(20, 5))

                img_label = ctk.CTkLabel(self.mask_frame, image=images_tk[i], text="")
                img_label.grid(row=row * 2 + 1, column=col, padx=20, pady=(0, 20))

        def display_next(self, destroy=True):
            if not self.images:
                return
            self.current_img_index = (self.current_img_index + 1) % len(self.images)
            self._update_display(destroy)

        def display_prev(self, destroy=True):
            if not self.images:
                return
            self.current_img_index = (self.current_img_index - 1) % len(self.images)
            self._update_display(destroy)

        def _update_display(self, destroy):
            image_path = self.images[self.current_img_index]
            

            img_slice1 = tensor_to_ctk_image(preprocess_single_image(image_path, slice=0))
            preds = make_predictions(image_path)
            img_slice2 = tensor_to_ctk_image(preprocess_single_image(image_path, slice=preds.shape[3] // 2))
            print("preds shape:", preds.shape)
            print(image_path)
            pred1 = tensor_to_ctk_image(preds[:, :, 0, 0])
            pred2 = tensor_to_ctk_image(preds[:, :, 0, preds.shape[3] // 2])

            self.images_tk = [img_slice1, pred1, img_slice2, pred2]
            self.display(self.images_tk, destroy=destroy)
        
        def download_masks(self):
            if self.mask_directory_chosen.get():
                for input_path in self.images:
                    pred_mask = make_predictions(input_path) 
                    file_name = os.path.basename(input_path)
                    if pred_mask.ndim == 4 and pred_mask.shape[2] == 1:
                        pred_mask = np.squeeze(pred_mask, axis=2)  # (H, W, D)

                    elif pred_mask.ndim != 3:
                        raise ValueError(f"Unexpected shape for prediction mask: {pred_mask.shape}")
                    output_filename = file_name.replace(".nii", "") + "_Mask.nii"
                    output_path = os.path.join(self.mask_directory_chosen.get(), output_filename)

                    ref_nii = nib.load(input_path)
                    original_shape = ref_nii.get_fdata().shape
                    resized_mask = []
                    for i in range(pred_mask.shape[2]):
                        slice_i = tf.image.resize(
                            pred_mask[:, :, i][..., np.newaxis],
                            (original_shape[0], original_shape[1]),
                            method='nearest').numpy().squeeze()
                        resized_mask.append(slice_i)

                    resized_mask = np.stack(resized_mask, axis=-1).astype(np.uint8)
                    affine = ref_nii.affine
                    header = ref_nii.header
                    pred_mask = pred_mask.astype(np.uint8)
                    pred_nii = nib.Nifti1Image(resized_mask, affine, header)
                    nib.save(pred_nii, output_path)
                    print(f"Saved predicted mask: {output_path}")
            else:
                messagebox.showerror("Error", "No directory chosen")

    # Variables
    directory_chosen = tk.StringVar(value='No directory chosen.')
    volume_download_file = tk.StringVar(value='No download file chosen.')

    image_height = tk.StringVar()
    image_width = tk.StringVar()
    image_depth = tk.StringVar()
    volume = tk.DoubleVar(value=0.0)

    # Title label
    address_label = ctk.CTkLabel(window, text="Please Add The MRI Images:", font=ctk.CTkFont(size=18, weight="bold"), fg_color=None,text_color="#333333")
    address_label.pack(pady=(15, 5))

    # Directory selection frame
    directory_frame = ctk.CTkFrame(window, fg_color=background_color, border_width=0)
    directory_frame.pack(pady=10, fill='x', padx=20)

    path_button = ctk.CTkButton(directory_frame,text="Choose Directory",command=choose_directory,corner_radius=12,fg_color='#4CAF50',hover_color="#45a049",text_color='white',font=ctk.CTkFont(size=14, weight="bold"),width=150)
    path_button.pack(side='left', pady=5, padx=(0, 20))

    directory_label = ctk.CTkLabel(directory_frame, 
                                textvariable=directory_chosen, 
                                fg_color="white",
                                text_color="#555555",
                                anchor='w',
                                corner_radius=6,
                                width=400)
    directory_label.pack(side='left', pady=10, fill='x', expand=True)

    # Helper to create labeled entry
    def create_labeled_entry(parent, label_text, variable):
        frame = ctk.CTkFrame(parent, fg_color=background_color, border_width=0)
        frame.pack(side='left', padx=10)
        label = ctk.CTkLabel(frame, text=label_text, text_color="#333333", font=ctk.CTkFont(size=14))
        label.pack(pady=5)
        entry = ctk.CTkEntry(frame, textvariable=variable, width=80, font=ctk.CTkFont(size=14), corner_radius=6)
        entry.pack(pady=5)
        return entry

    # Dimensions frame 1 (height, width)
    dimension_frame1 = ctk.CTkFrame(window, fg_color=background_color, border_width=0)
    dimension_frame1.pack(pady=10, fill='x', padx=20)

    image_height_entry = create_labeled_entry(dimension_frame1, "Image Length (mm):", image_height)
    image_width_entry = create_labeled_entry(dimension_frame1, "Image Width (mm):", image_width)

    # Dimensions frame 2 (depth)

    image_depth_entry = create_labeled_entry(dimension_frame1, "Image Depth (mm):", image_depth)

    # Buttons frame
    buttons_frame = ctk.CTkFrame(window, fg_color=background_color, border_width=0)
    buttons_frame.pack(pady=20, fill='x', padx=20)

    choose_download_path = ctk.CTkButton(buttons_frame,
                                        text="Choose Excel File",
                                        command=choose_download_file,
                                        corner_radius=12,
                                        fg_color='#2196F3',
                                        hover_color="#1976D2",
                                        text_color='white',
                                        font=ctk.CTkFont(size=16, weight="bold"),
                                        width=180)
    choose_download_path.pack(side='left', padx=10)

    download_chosen = ctk.CTkLabel(buttons_frame, 
                                textvariable=volume_download_file,
                                fg_color="white",
                                text_color="#555555",
                                anchor='w',
                                corner_radius=6,
                                width=400)
    download_chosen.pack(side='left', padx=10, fill='x', expand=True)

    download_volumes_btn = ctk.CTkButton(buttons_frame,
                                        text="Download Volumes",
                                        command=download_volumes,
                                        corner_radius=12,
                                        fg_color='#f44336',
                                        hover_color="#d32f2f",
                                        text_color='white',
                                        font=ctk.CTkFont(size=16, weight="bold"),
                                        width=180)
    download_volumes_btn.pack(side='left', padx=10)

    image_shower = ctk.CTkButton(buttons_frame,
                                text="Show Masks",
                                command=lambda: ImageDisplayer(window, background_color) if directory_chosen.get() else messagebox.showerror("Error", "Please choose a directory first"),
                                corner_radius=12,
                                fg_color='#4CAF50',
                                hover_color="#70db70",
                                text_color='white',
                                font=ctk.CTkFont(size=16, weight="bold"),
                                width=140)
    image_shower.pack(side='left', padx=10)

    window.mainloop()

if __name__ == '__main__':
    start()




import tkinter as tk
from tkinter import filedialog, messagebox, colorchooser, simpledialog
import numpy as np
import pandas as pd
import fitz  # PyMuPDF
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import re
from PIL import Image, ImageDraw, ImageTk
import io
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.utils import ImageReader
from sklearn.decomposition import PCA, FactorAnalysis


def apply_dark_plot_style():
    plt.style.use("dark_background")
    plt.rcParams.update({
        "text.color": "#00FF00",
        "axes.labelcolor": "#00FF00",
        "xtick.color": "#00FF00",
        "ytick.color": "#00FF00",
        "axes.edgecolor": "#00FF00",
        "figure.facecolor": "#000000",
        "axes.facecolor": "#000000"
    })


class PDFTableAnalyzer:
    def __init__(self):
        self.pdf_text = ""
        self.df = None

    def load_pdf_text(self, file_path):
        try:
            doc = fitz.open(file_path)
            self.pdf_text = "\n".join(page.get_text() for page in doc)
            doc.close()
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Could not open PDF: {e}")
            return False

    def extract_table(self):
        lines = [re.sub(r"\s+", " ", line.strip()) for line in self.pdf_text.splitlines() if line.strip()]
        if not lines:
            return None

        headers = lines[0].split()
        table = []

        for line in lines[1:]:
            columns = line.split()
            if len(columns) == len(headers):
                row = []
                for val in columns:
                    try:
                        row.append(float(val))
                    except ValueError:
                        row.append(np.nan)
                table.append(row)

        if not table:
            return None

        self.df = pd.DataFrame(table, columns=headers)
        return self.df

    def compute_correlation(self):
        if self.df is not None:
            methods = ['pearson', 'kendall', 'spearman']
            return {method: self.df.corr(method=method) for method in methods}
        return None

    def generate_scatter_plot(self, x_col, y_col):
        if self.df is None or x_col not in self.df.columns or y_col not in self.df.columns:
            return None

        x = self.df[x_col]
        y = self.df[y_col]

        apply_dark_plot_style()
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=x, y=y, color="#00FF00", label="Data Points")
        plt.grid(True, linestyle='--', alpha=0.7)

        coef = np.polyfit(x, y, 1)
        poly_fn = np.poly1d(coef)
        plt.plot(x, poly_fn(x), color="#FF00FF", linewidth=2, label=f"Best Fit: y={coef[0]:.2f}x + {coef[1]:.2f}")

        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f"{y_col} vs {x_col} with Best Fit Line")
        plt.legend()
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='PNG')
        buf.seek(0)
        plt.close()

        return buf

    def run_pca(self):
        if self.df is not None:
            df_clean = self.df.dropna()
            apply_dark_plot_style()
            pca = PCA(n_components=2)
            components = pca.fit_transform(df_clean)
            plt.figure(figsize=(8, 6))
            plt.scatter(components[:, 0], components[:, 1], c="#00FF00")
            plt.title("PCA Result")
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format="PNG")
            buf.seek(0)
            plt.close()
            return buf
        return None

    def run_afdm(self):
        if self.df is not None:
            df_clean = self.df.dropna()
            apply_dark_plot_style()
            fa = FactorAnalysis(n_components=2)
            components = fa.fit_transform(df_clean)
            plt.figure(figsize=(8, 6))
            plt.scatter(components[:, 0], components[:, 1], c="#00FF00")
            plt.title("Factor Analysis Result")
            plt.xlabel("F1")
            plt.ylabel("F2")
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format="PNG")
            buf.seek(0)
            plt.close()
            return buf
        return None


class DrawingBoard(tk.Canvas):
    def __init__(self, master, width=600, height=300, **kwargs):
        super().__init__(master, width=width, height=height, bg="#000000", **kwargs)
        self.bind("<ButtonPress-1>", self.start_draw)
        self.bind("<B1-Motion>", self.draw)
        self.color = "#00FF00"
        self.start_x = None
        self.start_y = None

        self.image = Image.new("RGB", (width, height), "black")
        self.draw_image = ImageDraw.Draw(self.image)

    def choose_color(self):
        color_code = colorchooser.askcolor(title="Choose a color")
        if color_code and color_code[1]:
            self.color = color_code[1]

    def start_draw(self, event):
        self.start_x = event.x
        self.start_y = event.y

    def draw(self, event):
        if self.start_x and self.start_y:
            self.create_line(self.start_x, self.start_y, event.x, event.y, fill=self.color, width=2)
            self.draw_image.line([self.start_x, self.start_y, event.x, event.y], fill=self.color, width=2)
            self.start_x = event.x
            self.start_y = event.y

    def save_as_image(self):
        output = io.BytesIO()
        self.image.save(output, format="PNG")
        output.seek(0)
        return output


class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PDF Analyzer - Hacker Mode")
        self.configure(bg="#000000")
        self.geometry("1200x900")

        self.analyzer = PDFTableAnalyzer()
        self.scatter_buf = None
        self.scatter_info = ("", "")

        self.create_widgets()

    def create_widgets(self):
        toolbar = tk.Frame(self, bg="#111111")
        toolbar.pack(side=tk.TOP, fill=tk.X)

        style_button = {
            "bg": "#111111",
            "fg": "#00FF00",
            "font": ("Courier", 10, "bold"),
            "activebackground": "#00FF00",
            "activeforeground": "#000000",
            "relief": tk.GROOVE,
            "bd": 1
        }

        for text, cmd in [
            ("Open PDF", self.open_pdf),
            ("Extract Table", self.extract_table),
            ("Choose Pen Color", self.choose_color),
            ("Run PCA / Factor", self.run_dimensionality_reduction),
            ("Save Report", self.save_report)
        ]:
            tk.Button(toolbar, text=text, command=cmd, **style_button).pack(side=tk.LEFT, padx=5, pady=5)

        self.text_area = tk.Text(self, wrap=tk.WORD, height=20,
                                 bg="#000000", fg="#00FF00",
                                 insertbackground="#00FF00",
                                 font=("Courier", 10))
        self.text_area.pack(fill=tk.BOTH, expand=True)

        self.observations = tk.Text(self, wrap=tk.WORD, height=5,
                                    bg="#000000", fg="#00FF00",
                                    insertbackground="#00FF00",
                                    font=("Courier", 10))
        self.observations.pack(fill=tk.X, padx=10, pady=5)
        self.observations.insert(tk.END, "Write observations here...")

        self.drawing_board = DrawingBoard(self, width=800, height=300, highlightbackground="#00FF00")
        self.drawing_board.pack(pady=10)

    def open_pdf(self):
        file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
        if file_path and self.analyzer.load_pdf_text(file_path):
            self.text_area.delete(1.0, tk.END)
            self.text_area.insert(tk.END, self.analyzer.pdf_text)
            messagebox.showinfo("Success", "PDF Loaded Successfully!")

    def extract_table(self):
        if not self.analyzer.pdf_text:
            messagebox.showwarning("Warning", "No PDF loaded yet!")
            return

        df = self.analyzer.extract_table()
        if df is None:
            messagebox.showinfo("Info", "No valid table detected.")
            return

        self.text_area.delete(1.0, tk.END)
        self.text_area.insert(tk.END, "Extracted Table:\n")
        self.text_area.insert(tk.END, df.to_string(index=False))

        correlations = self.analyzer.compute_correlation()
        if correlations:
            for method, corr in correlations.items():
                self.text_area.insert(tk.END, f"\n\nCorrelation Matrix ({method.capitalize()}):\n")
                self.text_area.insert(tk.END, corr.to_string())

        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) >= 2:
            x_col = simpledialog.askstring("Scatter Plot", f"Choose X-axis column:\n{list(numeric_cols)}")
            y_col = simpledialog.askstring("Scatter Plot", f"Choose Y-axis column:\n{list(numeric_cols)}")
            if x_col in numeric_cols and y_col in numeric_cols:
                self.scatter_buf = self.analyzer.generate_scatter_plot(x_col, y_col)
                self.scatter_info = (x_col, y_col)
            else:
                messagebox.showwarning("Warning", "Invalid columns selected.")

    def run_dimensionality_reduction(self):
        method = simpledialog.askstring("Choose Method", "Enter 'PCA' or 'AFDM' for dimensionality reduction")
        if method is None:
            return

        method = method.strip().lower()
        if method == "pca":
            self.scatter_buf = self.analyzer.run_pca()
            self.scatter_info = ("PC1", "PC2")
        elif method == "afdm":
            self.scatter_buf = self.analyzer.run_afdm()
            self.scatter_info = ("F1", "F2")
        else:
            messagebox.showwarning("Invalid Input", "Please enter 'PCA' or 'AFDM'.")

    def choose_color(self):
        self.drawing_board.choose_color()

    def save_report(self):
        save_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF Files", "*.pdf")])
        if not save_path:
            return

        try:
            sig_buf = self.drawing_board.save_as_image()
            c = pdf_canvas.Canvas(save_path, pagesize=A4)
            width, height = A4

            c.setFont("Courier", 10)
            textobject = c.beginText(40, height - 40)
            text_lines = self.text_area.get(1.0, tk.END).splitlines()
            for line in text_lines:
                textobject.textLine(line)
                if textobject.getY() < 100:
                    c.drawText(textobject)
                    c.showPage()
                    textobject = c.beginText(40, height - 40)
            c.drawText(textobject)

            if self.scatter_buf:
                scatter_reader = ImageReader(self.scatter_buf)
                c.showPage()
                c.drawString(50, height - 50, f"Scatter Plot: {self.scatter_info[1]} vs {self.scatter_info[0]}")
                c.drawImage(scatter_reader, 50, 150, width=500, preserveAspectRatio=True, mask='auto')

            observations = self.observations.get(1.0, tk.END).strip()
            if observations:
                c.showPage()
                c.setFont("Courier", 12)
                c.drawString(50, height - 50, "Observations:")
                text_obs = c.beginText(50, height - 70)
                for line in observations.splitlines():
                    text_obs.textLine(line)
                    if text_obs.getY() < 100:
                        c.drawText(text_obs)
                        c.showPage()
                        text_obs = c.beginText(50, height - 50)
                c.drawText(text_obs)

            sig_reader = ImageReader(sig_buf)
            c.showPage()
            c.drawString(50, height - 50, "Signature:")
            c.drawImage(sig_reader, 50, 150, width=400, preserveAspectRatio=True, mask='auto')

            c.save()
            messagebox.showinfo("Success", "Report saved successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save report: {e}")


if __name__ == "__main__":
    app = Application()
    app.mainloop()

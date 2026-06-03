import tkinter as tk
import pmain3
import cv2 # used for video/image processing and showing frames
from PIL import Image, ImageTk
from datetime import datetime
import subprocess
import platform
from tkinter import ttk
import os
import threading
from PIL import Image, ImageTk
import serial
import serial.tools.list_ports
from pathlib import Path
import subprocess
import sys
import threading

# ensure project root is on sys.path so sibling packages import correctly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "symbols"

from synchronizeData import synchronize_and_merge


class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tipwindow = None
        self.delay = 500  # delay in milliseconds
        self.after_id = None
        

        widget.bind("<Enter>", self.schedule_tip)
        widget.bind("<Leave>", self.hide_tip)

    def schedule_tip(self, event=None):
        self.after_id = self.widget.after(self.delay, self.show_tip)

    def show_tip(self, event=None):
        if self.tipwindow or not self.text:
            return

        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5

        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")

        label = tk.Label(
            tw,
            text=self.text,
            justify="left",
            bg="#111827",
            fg="white",
            relief="solid",
            borderwidth=1,
            font=("Arial", 10),
            wraplength=260
        )
        label.pack(ipadx=8, ipady=4)

    def hide_tip(self, event=None):
        if self.after_id:
            self.widget.after_cancel(self.after_id)
            self.after_id = None

        if self.tipwindow:
            self.tipwindow.destroy()
            self.tipwindow = None


#------------------------------------------Basic structure------------------------------------------
BG_MAIN        = "#E5E7EB"   # background
BG_PANEL       = "#D4D8DE"   # panels and frames
BG_HEADER      = "#5B677A"   # headers and top bar
BG_SOFT        = "#F1F5F9"   # secondary surfaces
BG_DARK        = "#0F172A"   # video 

TEXT_PRIMARY   = "#0F172A"
TEXT_SECONDARY = "#475569"
TEXT_INVERTED  = "#FFFFFF"

STATUS_OK      = "#22C55E"
STATUS_WARN    = "#F59E0B"
STATUS_ERROR   = "#EF4444"

window = tk.Tk()
window.title("Occupancy Grid Review Frame")
window.configure(bg=BG_MAIN)
window.resizable(True, True)

window.geometry("1200x900+0+0")

def createFrame(parentFrame, _bg, _highlightbackground, _x, _y, _width, _height):
    frame = tk.Frame(parentFrame, bg=_bg, highlightthickness=1, highlightbackground=_highlightbackground)
    frame.place(x=_x, y=_y, width=_width, height=_height)
    return frame

container = createFrame(window, BG_PANEL, BG_DARK, 48, 160, 1100, 520)

YoloCameraFrame = tk.Frame(container, bg=BG_PANEL)
reviewFrame = tk.Frame(container, bg=BG_PANEL)


#------------------------------------------Methods for GUI------------------------------------------
def open_settings_window():
    popUpSettings = tk.Toplevel(window)
    popUpSettings.title("Settings")
    popUpSettings.geometry("800x500")
    popUpSettings.configure(bg=BG_MAIN)
    settingsFrame = createFrame(popUpSettings, BG_PANEL, BG_DARK, 8, 17, 780, 470)
    
    aboutUsFrame = createFrame(settingsFrame, BG_PANEL, BG_DARK, 520, 5, 250, 455)
    create_Label_Using_Config(aboutUsFrame, "top", 3, "About us", "none", BG_PANEL, 30, TEXT_PRIMARY)
    text = "We are a team of 8 students who developed this product as our thesis project. The system collects indoor radar data and uses machine learning to estimate occupancy. A camera running YOLOv13 provides ground-truth data synchronized with the radar for training. Once trained, the radar alone can estimate the number of people in real time, with the camera used only for optional validation."
    create_Label_Using_Place(aboutUsFrame, BG_PANEL, 5, 70, text, 12, 240, TEXT_PRIMARY)
    
    specificationsFrameOut = createFrame(settingsFrame, BG_PANEL, BG_DARK, 7, 5, 500, 270)
    create_Label_Using_Config(specificationsFrameOut, "top", 3, "Recomended specifications", "none", BG_PANEL, 30, TEXT_PRIMARY)
    container = createFrame(specificationsFrameOut, BG_PANEL, BG_DARK, 7, 90, 480, 170)

    hardwareFrame = tk.Frame(container, bg=BG_PANEL)
    radarFrame = tk.Frame(container, bg=BG_PANEL)
    YoloCameraFrame = tk.Frame(container, bg=BG_PANEL)


    for frame in (radarFrame, hardwareFrame, YoloCameraFrame): # stack all frames on top of each other
        frame.place(relwidth=1, relheight=1)

    create_btn(8, 55, 100, 30, "Radar", specificationsFrameOut, True, radarFrame, "no method call")
    create_btn(120, 55, 100, 30, "Hardware", specificationsFrameOut, True, hardwareFrame, "no method call")
    create_btn(232, 55, 100, 30, "Camera", specificationsFrameOut, True, YoloCameraFrame, "no method call")

    create_Label_Using_Config(radarFrame, "top", 3, "Radar", "none", BG_PANEL, 20, TEXT_PRIMARY)
    txtRadar = "Radar name: HMMD mmWave \nRadar chip: S3KM1110 \nCommunication method: UART \nRadar baud rate: 115200 \nFrame rate: 10 Hz"
    create_Label_Using_Place(radarFrame, BG_PANEL, 10, 40, txtRadar, 15, 400, TEXT_PRIMARY)

    create_Label_Using_Config(hardwareFrame, "top", 3, "Hardware", "none", BG_PANEL, 20, TEXT_PRIMARY)
    txtHardware = "Hardware name: Jetson Nano\nUART-pins: 40-pin header\nBaud rate: 115200"
    create_Label_Using_Place(hardwareFrame, BG_PANEL, 10, 40, txtHardware, 15, 400, TEXT_PRIMARY)

    create_Label_Using_Config(YoloCameraFrame, "top", 3, "Camera", "none", BG_PANEL, 20, TEXT_PRIMARY)
    txtCamera = "Processor: Dual-core Tensilica LX6, 240 MHz\nMemory: 4 MB PSRAM, 32 Mbit flash\nWireless: Wi-Fi 802.11 b/g/n, Bluetooth 4.2\nCamera: OV2640, 2 MP, JPEG\nGPIO: UART, SPI, I²C, PWM"
    create_Label_Using_Place(YoloCameraFrame, BG_PANEL, 10, 40, txtCamera, 15, 400, TEXT_PRIMARY)

    otherFrame = createFrame(settingsFrame, BG_PANEL, BG_DARK, 7, 290, 500, 170)
    create_Label_Using_Config(otherFrame, "top", 3, "Other", "none", BG_PANEL, 30, TEXT_PRIMARY)
    
    create_btn(180, 70, 130, 30, "Create folder", otherFrame, False, "none", createFolder)

def createFolder():
    os.makedirs("results_Occupancy", exist_ok=True) 

# The subframe argument should always be false for the buttons.
# The only 3 subframes are the steps frames: YoloCameraFrame and reviewFrame.
def create_btn(Btnx, Btny, Btnwidth, Btnheight, btnText, parentFrame, isSubFrame, subFrame, methodCall):
    style = ttk.Style()
    style.theme_use("clam")

    style.configure(
    "CustomButton.TButton", background=BG_HEADER,
    foreground=TEXT_INVERTED,
    borderwidth=2, focusthickness=1,
    focuscolor="#7D8EE3", padding=(10, 0),
    anchor="center",font=("Arial", 12)
    )


    if(isSubFrame):
        btn = ttk.Button(parentFrame, text=btnText,style="CustomButton.TButton", command=lambda: show(subFrame))
    else:
        btn = ttk.Button(parentFrame, text=btnText,style="CustomButton.TButton", command=methodCall)
    
    btn.place(x=Btnx, y=Btny, width=Btnwidth, height=Btnheight)
    return btn

def create_TextBox(x_, y_, width_, height_, parentFrame):
    textbox = tk.Text(parentFrame, width=width_, height=height_, bg=BG_DARK, fg="white", font=("Arial", 12))
    textbox.insert("1.0", "")
    textbox.config(state="disabled")      # gör textbox read-only
    textbox.place(x=x_, y=y_)
    return textbox

def create_Label_Using_Config(parentFrame, packingSide, _padx, lblText, defaultText, _bg, fontSize, color):
        label = tk.Label(parentFrame, bg=_bg, fg=color, font=("Arial", fontSize))
        label.pack(side=packingSide, padx=_padx)
        label.config(text=lblText or defaultText)

def create_Label_Using_Place(parentframe, _bg, _x, _y, _text, textSize, _wraplength, color):
    label = tk.Label(parentframe, bg=_bg, fg=color, font=("Arial", textSize), wraplength=_wraplength)
    label.place(x=_x, y=_y)
    label.config(text=_text)
    return label

def create_btn_with_image(picURL, imgW, imgH, parentFrame, methodCall, _x, _y):
    pic_path = Path(picURL)
    if not pic_path.is_absolute():
        # allow callers to pass either "change.png" or "symbols/change.png"
        if pic_path.parts and pic_path.parts[0] == "symbols":
            pic_path = ASSETS_DIR / Path(*pic_path.parts[1:])
        else:
            pic_path = ASSETS_DIR / pic_path

    img = Image.open(pic_path)
    img = img.resize((imgW, imgH), Image.LANCZOS)   # skala ner utan att förstöra kvalitet
    changePic = ImageTk.PhotoImage(img)

    btn = tk.Button(
        parentFrame,
        image=changePic,
        borderwidth=0,
        highlightthickness=0,
        bg=parentFrame["bg"],
        activebackground=parentFrame["bg"], command=methodCall
    )
    btn.image = changePic #keeping the reference
    btn.place(x=_x, y=_y, width=imgW, height=imgH)
    return btn

def get_wifi_name():
    os_name = platform.system()

    if os_name == "Windows": # Windows
        command = ["netsh", "wlan", "show", "interfaces"] #kommandot för att visa WiFi info
        result = subprocess.run(command, capture_output=True, text=True) #kör kommandot. "capture_output=True" fångar upp outputen. "text=True" gör att outputen blir en sträng istället för bytes
        
        for line in result.stdout.split('\n'): #går igenom varje rad i outputen
            if "SSID" in line and "BSSID" not in line: #letar efter raden som innehåller "SSID"
                return line.split(":")[1].strip() #delar upp raden vid ":" och tar den andra delen (SSID-namnet), tar bort eventuella extra mellanslag

    elif os_name == "Darwin":  # macOS
        command = ["/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport", "-I"]
        result = subprocess.run(command, capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if " SSID" in line:
                return line.split(":")[1].strip()

    elif os_name == "Linux": # Linux
        command = ["iwgetid", "-r"]
        result = subprocess.run(command, capture_output=True, text=True)
        return result.stdout.strip()

    return None

def show(frame):
    frame.tkraise()

# Creates a reusable GUI control for live-updating detection parameters at runtime
def create_live_parameter_control(tb_x, tb_y, tb_w, tb_h, tb_parentFrame, lbl_x, lbl_y, lblText, btn_x, btn_y, notesError, variableChange, notesVariable):
    tb = create_TextBox(tb_x, tb_y * 1.5, tb_w, tb_h, tb_parentFrame)
    lbl = create_Label_Using_Place(tb_parentFrame, BG_PANEL, lbl_x, lbl_y * 1.5, lblText, 12, 200, TEXT_PRIMARY)
    btn = create_btn_with_image("symbols/change.png", 23, 23, tb_parentFrame, lambda: update_live_parameter(tb, notesError, variableChange, notesVariable), btn_x, btn_y * 1.5)
    return tb, lbl

def update_live_parameter(tb, notesError, setter, notesVariable):
    try:
        value_str = tb.get("1.0", tk.END).strip()
        value = float(value_str)
    except ValueError:
        display_status_text(notesError)
        return

    setter(value)
    display_status_text(f"{notesVariable} changed to: {value}")


#Hämtar det nuvarande värdet från pmain3, bara för visning.
def set_value_TB(value, tb):
    tb.config(state="normal")
    tb.delete("1.0", tk.END)
    tb.insert("1.0", str(value))
    tb.config(state="normal")

def update_fps(fps):
    fps_var.set(f"FPS: {fps:.1f}")

def update_preview(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    imgtk = ImageTk.PhotoImage(image=img)

    preview_label.imgtk = imgtk
    preview_label.config(image=imgtk)


def set_stable_values():
    pmain3.MIN_CONFIDENCE = 0.35
    pmain3.MIN_AREA = 100
    pmain3.MAX_AREA = float('inf')
    pmain3.ASPECT_RATIO_MIN = 0.25
    pmain3.ASPECT_RATIO_MAX = 4.0
    pmain3.VIDEO_FPS_VALUE = 5
    pmain3.DRAW_BOXES = True

    set_value_TB(pmain3.MIN_CONFIDENCE, confTB[0])
    set_value_TB(pmain3.MIN_AREA, MIN_AREA_TB[0])
    set_value_TB(pmain3.MAX_AREA, MAX_AREA_TB[0])
    set_value_TB(pmain3.ASPECT_RATIO_MIN, ASPECT_RATIO_MIN_TB[0])
    set_value_TB(pmain3.ASPECT_RATIO_MAX, ASPECT_RATIO_MAX_TB[0])
    set_value_TB(pmain3.VIDEO_FPS_VALUE, VIDEO_FPS[0])

    draw_boxes_var.set(True)

    display_status_text("Parameters set to stable")

def set_sensitive_values():
    pmain3.MIN_CONFIDENCE = 0.30
    pmain3.MIN_AREA = 80
    pmain3.MAX_AREA = float('inf')
    pmain3.ASPECT_RATIO_MIN = 0.20
    pmain3.ASPECT_RATIO_MAX = 4.5
    pmain3.VIDEO_FPS_VALUE = 8
    pmain3.DRAW_BOXES = True

    set_value_TB(pmain3.MIN_CONFIDENCE, confTB[0])
    set_value_TB(pmain3.MIN_AREA, MIN_AREA_TB[0])
    set_value_TB(pmain3.MAX_AREA, MAX_AREA_TB[0])
    set_value_TB(pmain3.ASPECT_RATIO_MIN, ASPECT_RATIO_MIN_TB[0])
    set_value_TB(pmain3.ASPECT_RATIO_MAX, ASPECT_RATIO_MAX_TB[0])
    set_value_TB(pmain3.VIDEO_FPS_VALUE, VIDEO_FPS[0])

    draw_boxes_var.set(True)

    display_status_text("Parameters set to sensitive")


def thread_start_radar():
    radar_thread = threading.Thread(
        target=start_radar,
        daemon=True
    )
    radar_thread.start()
    

def start_radar():
    global radar_process

    repoB_dir = Path("../OccupancyRadar").resolve()
    time_script = repoB_dir / "time.py"
    run_script = repoB_dir / "run.py"

    # 1. Kör time.py (kort script)
    subprocess.run(
        [sys.executable, str(time_script)],
        check=True
    )

    # 2. Starta run.py (långlivad process)
    radar_process = subprocess.Popen(
        [sys.executable, str(run_script)]
    )

    display_status_text("Radar started")
  
def start_recording():
    stop_event.clear()  # säkerställ att vi inte är i stopp-läge

    threading.Thread(
        target=pmain3.run,
        args=(update_preview, display_notes_text, stop_event, update_fps),
        daemon=True
    ).start()

    display_status_text("Recording started")
    thread_start_radar()


# Draw boxes
draw_boxes_var = tk.BooleanVar(value=pmain3.DRAW_BOXES)

def toggle_draw_boxes():
    pmain3.DRAW_BOXES = draw_boxes_var.get()
    display_status_text(f"Draw boxes set to: {pmain3.DRAW_BOXES}")

tk.Checkbutton(
    YoloCameraFrame,
    text="Show bounding boxes",
    variable=draw_boxes_var,
    command=toggle_draw_boxes,
    bg=BG_PANEL,
    font=("Arial", 12)
).place(x=5, y=350)

stop_event = threading.Event()

from pathlib import Path

def getCSVPaths():
    BASE_DIR = Path(__file__).resolve().parent

    # Radar CSV (OccupancyRadar)
    DATA_DIR = (BASE_DIR / "../OccupancyRadar/Data").resolve()
    radar_csv = max(
        DATA_DIR.glob("*.csv"),
        key=lambda p: p.stat().st_mtime
    )

    # Kamera CSV (OccupancyDetection/results)
    RESULTS_DIR = BASE_DIR / "results"
    latest_results_dir = max(
        (p for p in RESULTS_DIR.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime
    )

    camera_csv = next(latest_results_dir.glob("*.csv"))

    # Output
    output_csv = latest_results_dir / "merged.csv"

    synchronize_and_merge(camera_csv, radar_csv, output_csv)

def stop_radar():
    global radar_process
    if radar_process is not None:
        radar_process.terminate()
        radar_process = None
        display_status_text("Radar stopped")

def stop_recording():
    stop_event.set()
    stop_radar()
    preview_label.config(image="")
    clear_notes()
    display_status_text("Recording stopped")
    getCSVPaths()

def reset_values():
    pmain3.MIN_CONFIDENCE = 0.35
    pmain3.MIN_AREA = 100
    pmain3.MAX_AREA = float('inf')
    pmain3.ASPECT_RATIO_MIN = 0.25
    pmain3.ASPECT_RATIO_MAX = 4.0
    pmain3.VIDEO_FPS_VALUE = 5
    pmain3.DRAW_BOXES = True

    set_value_TB(pmain3.MIN_CONFIDENCE, confTB[0])
    set_value_TB(pmain3.MIN_AREA, MIN_AREA_TB[0])
    set_value_TB(pmain3.MAX_AREA, MAX_AREA_TB[0])
    set_value_TB(pmain3.ASPECT_RATIO_MIN, ASPECT_RATIO_MIN_TB[0])
    set_value_TB(pmain3.ASPECT_RATIO_MAX, ASPECT_RATIO_MAX_TB[0])
    set_value_TB(pmain3.VIDEO_FPS_VALUE, VIDEO_FPS[0])

    draw_boxes_var.set(True)

    display_status_text("Parameters reset to default")

def display_notes_text(text):
    notes_textbox.config(state="normal")
    notes_textbox.insert(tk.END, text + "\n")
    notes_textbox.see(tk.END)
    notes_textbox.config(state="disabled")

def display_status_text(text):
    status_textbox.config(state="normal")
    status_textbox.insert(tk.END, text + "\n")
    status_textbox.see(tk.END)
    status_textbox.config(state="disabled")

def clear_notes():
    notes_textbox.config(state="normal")
    notes_textbox.delete("1.0", tk.END)
    notes_textbox.config(state="disabled")

#This method fins the correct COM-port. It iterates
#through all COM ports in the computer and searches
#thorugh typical USB serial devices (Arduino/ESP32)
#it then returns the port
def find_port():
    ports = serial.tools.list_ports.comports()

    for port in ports:
        if "USB" in port.description or "Arduino" in port.description or "CP210" in port.description:
            return port.device
    return None  




#------------------------------------------Building GUI------------------------------------------

#Hero frame
heroFrame = createFrame(window, BG_HEADER, BG_HEADER, 0, 0, 1200, 50)

#Date label
date_label = tk.Label(heroFrame, bg=BG_HEADER, fg=TEXT_INVERTED, font=("Arial", 14))
date_label.place(x=550, y=10)

def update_date():
    current_date = datetime.now().strftime("%Y-%m-%d")
    date_label.config(text=current_date) #byter ut radarns text med det nya datumet

update_date()

#Settings button
img = Image.open(ASSETS_DIR / "settings.png")
img = img.resize((40, 40), Image.LANCZOS)   # skala ner utan att förstöra kvalitet
settingsPic = ImageTk.PhotoImage(img)

settingsBtn = tk.Button(
    heroFrame,
    image=settingsPic,
    borderwidth=0,
    highlightthickness=0,
    bg=heroFrame["bg"],
    activebackground=heroFrame["bg"], command=lambda: open_settings_window()
)
settingsBtn.image = settingsPic #keeping the reference
settingsBtn.place(x=3, y=5, width=40, height=40)

wifi_frame = createFrame(window, BG_HEADER, BG_HEADER, 1000, 2, 200, 45)

#WiFi image
imgWifi = Image.open(ASSETS_DIR / "wifi.webp")
imgWifi = imgWifi.resize((40, 40), Image.LANCZOS)   # skala ner utan att förstöra kvalitet
wifiPic = ImageTk.PhotoImage(imgWifi)

wifiPictureLabel = tk.Label(
    wifi_frame,
    image=wifiPic,
    borderwidth=0,
    highlightthickness=0,
    bg=wifi_frame["bg"],
    activebackground=wifi_frame["bg"]
)
wifiPictureLabel.pack(side="right")

create_Label_Using_Config(wifi_frame, "right", 0, get_wifi_name(), "No Wifi", BG_HEADER, 12, TEXT_INVERTED)


for frame in (reviewFrame, YoloCameraFrame): # stack all frames on top of each other
    frame.place(relwidth=1, relheight=1)

show(YoloCameraFrame)

create_btn(48, 120, 195, 35, "Data collection", window, True, YoloCameraFrame, "no method call")
create_btn(258, 120, 195, 35, "Evaluation", window, True, reviewFrame, "no method call")

notesFrame = createFrame(window, BG_PANEL, BG_DARK, 49, 690, 534, 182)
create_Label_Using_Place(notesFrame, BG_PANEL, 10, 5, "Notes", 20, 400, TEXT_PRIMARY)
notes_textbox = create_TextBox(12, 45, 57, 7, notesFrame)

statusFrame = createFrame(window, BG_PANEL, BG_DARK, 615, 690, 534, 182)
create_Label_Using_Place(statusFrame, BG_PANEL, 10, 5, "Status", 20, 400, TEXT_PRIMARY)
status_textbox = create_TextBox(12, 45, 57, 7, statusFrame)

#------------------------------------------REVIEW FRAME------------------------------------------
modelReviewFrame = createFrame(reviewFrame, BG_PANEL, BG_DARK, 7, 7, 650, 130)

create_Label_Using_Place(modelReviewFrame, BG_PANEL, 10, 10, "Model Review: Compare Radar", 20, 400, TEXT_PRIMARY)

create_btn(10, 70, 130, 30, "Run Review", modelReviewFrame, False, None, "no method call")

create_TextBox(220, 73, 5, 1.2, modelReviewFrame)
create_Label_Using_Place(modelReviewFrame, BG_PANEL, 170, 72, "MAE:", 12, 400, TEXT_PRIMARY)

create_TextBox(360, 73, 5, 1.2, modelReviewFrame)
create_Label_Using_Place(modelReviewFrame, BG_PANEL, 300, 72, "RMSE:", 12, 400, TEXT_PRIMARY)

create_TextBox(500, 73, 5, 1.2, modelReviewFrame)
create_Label_Using_Place(modelReviewFrame, BG_PANEL, 458, 72, "Corr:", 12, 400, TEXT_PRIMARY)

#-------------------------------------------YOLO FRAME------------------------------------------
YOLO_START_Y = 20
YOLO_GAP = 32

tk.Label(
    YoloCameraFrame,
    text="IP adress for camera stream",
    bg=BG_PANEL,
    fg="black",
    font=("Arial", 14, "bold")
).place(x=10, y=YOLO_START_Y)

ip_entry = tk.Entry(
    YoloCameraFrame,
    bg=BG_DARK,
    fg="white",
    disabledbackground=BG_DARK,
    disabledforeground="white",
    insertbackground="white",
    font=("Arial", 12),
    relief="flat"
)
ip_entry.place(x=280, y=YOLO_START_Y, width=200, height=28)

ip_entry.config(state="normal")
ip_entry.insert(tk.END, "169.254.41.208")
ip_entry.config(state="disabled")


preview_box = tk.Frame(YoloCameraFrame, bg=BG_DARK, highlightthickness=1)

preview_box.place(x=205, y=76, width=710, height=400)

preview_label = tk.Label(preview_box, bg="black")
preview_label.pack(fill="both", expand=True)

#CONFIDENCE
confTB = create_live_parameter_control(
    90, 60, 6, 1.2, YoloCameraFrame,
    5, 59, "Confidence",
    147, 59,
    "Invalid confidence value",
    lambda v: setattr(pmain3, "MIN_CONFIDENCE", v),
    "Confidence"
)
set_value_TB(pmain3.MIN_CONFIDENCE, confTB[0])

ToolTip(
    confTB[1],
    "Minimum confidence score required for a detection to be counted as a person.\n"
    "Higher values reduce false positives but may cause missed detections.\n"
    "Lower values increase sensitivity but may introduce noise."
)

#MIN_AREA CODE
MIN_AREA_TB = create_live_parameter_control(
    110, 90, 6, 1.2, YoloCameraFrame,
    5, 89, "Minimum area",
    163, 90,
    "Invalid minimum area value",
    lambda v: setattr(pmain3, "MIN_AREA", v),
    "Minimum area"
)
set_value_TB(pmain3.MIN_AREA, MIN_AREA_TB[0])

ToolTip(
    MIN_AREA_TB[1],
    "Minimum bounding box area in pixels.\n"
    "Used to filter out small or partial detections such as hands, legs, or background noise."
)

#MAX_AREA CODE
MAX_AREA_TB = create_live_parameter_control(
    115, 121, 6, 1.2, YoloCameraFrame,
    5, 119, "Maximum area",
    175, 122,
    "Invalid maximum area value",
    lambda v: setattr(pmain3, "MAX_AREA", v),
    "Maximum area"
)
set_value_TB(pmain3.MAX_AREA, MAX_AREA_TB[0])

ToolTip(
    MAX_AREA_TB[1],
    "Maximum allowed bounding box area in pixels. Helps prevent very large detections\n"
    "(e.g. merged objects or close-up false positives) from being counted"
)

#ASPECT RATIO MIN CODE
ASPECT_RATIO_MIN_TB = create_live_parameter_control(
    130, 150, 5, 1.2, YoloCameraFrame,
    5, 149, "Aspect Ratio Min",
    180, 150,
    "Invalid Aspect Ratio Min value",
    lambda v: setattr(pmain3, "ASPECT_RATIO_MIN", v),
    "Aspect Ratio Min"
)
set_value_TB(pmain3.ASPECT_RATIO_MIN, ASPECT_RATIO_MIN_TB[0])

ToolTip(
    ASPECT_RATIO_MIN_TB[1],
    "Minimum height-to-width ratio of a bounding box. Filters out detections that do not \nresemble the shape of a standing person."
)

# ASPECT RATIO MAX CODE
ASPECT_RATIO_MAX_TB = create_live_parameter_control(
    132, 180, 5, 1.2, YoloCameraFrame,
    5, 179, "Aspect Ratio Max",
    180, 180,
    "Invalid Aspect Ratio Max value",
    lambda v: setattr(pmain3, "ASPECT_RATIO_MAX", v),
    "Aspect Ratio Max"
)

ToolTip(
    ASPECT_RATIO_MAX_TB[1],
    "Maximum height-to-width ratio of a bounding box. Prevents extremely tall or narrow detections from being counted."
)

set_value_TB(pmain3.ASPECT_RATIO_MAX, ASPECT_RATIO_MAX_TB[0])

# VIDEO FPS
VIDEO_FPS = create_live_parameter_control(
    150, 210, 3, 1.2, YoloCameraFrame,
    5, 210, "Frames per second",
    180, 210,
    "Invalid FPS value",
    lambda v: setattr(pmain3, "VIDEO_FPS_VALUE", v),
    "FPS"
)

ToolTip(
    VIDEO_FPS[1],
    "Target processing frame rate. Lower values reduce CPU load. Higher values improve responsiveness but require more processing power."
)

set_value_TB(pmain3.VIDEO_FPS_VALUE, VIDEO_FPS[0])

# DISPLAY OF FPS
fps_var = tk.StringVar(value="FPS: -")

fps_label = tk.Label(
    YoloCameraFrame,
    textvariable=fps_var,
    bg=BG_PANEL,
    font=("Arial", 12)
)
fps_label.place(x=920, y=84)

stableModeBtn = create_btn(5, 390, 70, 30, "Stable", YoloCameraFrame, False, None, set_stable_values)
sensitiveModeBtn = create_btn(85, 390, 90, 30, "Sensitive", YoloCameraFrame, False, None, set_sensitive_values)

# Start recording, stop recording, and reset button
BTN_WIDTH = 160
BTN_HEIGHT = 35
BTN_GAP = 40
BTN_Y = 480

TOTAL_BTN_WIDTH = BTN_WIDTH * 3 + BTN_GAP * 2
START_X = (1080 - TOTAL_BTN_WIDTH) // 2

create_btn(START_X, BTN_Y, BTN_WIDTH, BTN_HEIGHT, "Stop recording", YoloCameraFrame, False, None, stop_recording)

create_btn(START_X + BTN_WIDTH + BTN_GAP, BTN_Y, BTN_WIDTH, BTN_HEIGHT, "Start recording", YoloCameraFrame, False,  None, start_recording)

create_btn(START_X + 2 * (BTN_WIDTH + BTN_GAP), BTN_Y, BTN_WIDTH, BTN_HEIGHT, "Reset values", YoloCameraFrame, False, None, reset_values)


window.mainloop()
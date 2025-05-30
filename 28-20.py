import georinex as gr
import hatanaka
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import xarray as xr
#import pymap3d as pd

import scipy as sc # for least squares fit and probably more
from scipy.optimize import leastsq #    ^

import requests # to get the most recent (most recent hourly) set of data
import datetime #   ^
import shutil #     ^

import os #for deleting downloaded file after calculations are done

today=datetime.datetime.today()
YYYY=today.year
HH=today.hour
DDD=today.timetuple().tm_yday
YY=YYYY-2000



EUR='EUR'

import tkinter as tk
from tkinter import simpledialog
from tkinter import messagebox
import threading
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# List of options just copied off the website, they probably work fine idk
'''
eur_options = [
    ('Eureka', 'EUR'), # Error 1 is C2W vs C2L
    ('Arctic Bay', 'ARC'), # Error 2 is presumably that station doesn't use those intervals to measure
    ('Arviat', 'ARV'), #
    ('Cambridge Bay❌1', 'CBB'),
    ('Kugluktu❌2', 'KUG'),
    ('Resolute❌1', 'RES'),
    ('Iqaluit❌1', 'IQA'),
    ('Ottawa❌2', 'OTT'),
    ('Pond Inlet❌2', 'PON'),
    ('Churchill❌2', 'CHU'),
    ('Coral Harbour❌2', 'COR'),
    ('Fort McMurray', 'MCM'), #
    ('City of Dawson', 'DAW'), #
    ('Ministik Lake', 'EDM'), #
    ('Grise Fiord❌2', 'GRI'),
    ('Gjoa Haven', 'GJO'), #
    ('Fort Simpson', 'FSI'), #
    ('Qikiqtarjuaq', 'QIK'), #
    ('Gilliam', 'GIL'), #
    ('Fort Smith', 'FSM'), #
    ('Rabbit Lake', 'RAB'), #
    ('Rankin Inlet', 'RAN'), #
    ('Taloyoak/Vernadsky Station', 'TAL'), #
    ('Whale Cove❌2', 'WHA'),
    ('Uapishka❌2', 'UAP'),
    ('Repulse Bay❌2', 'REP'),
    ('Sanikiluaq❌2', 'SAN'),
    ('Sachs Harbour', 'SAC'),
]
'''


eur_options = [
    ('Eureka', 'EUR', 4), # Error 1 is C2W vs C2L
    ('Arctic Bay', 'ARC', 3), # Error 2 is presumably that station doesn't use those intervals to measure
    ('Arviat', 'ARV', 3), # Number 3 is theoretically fine but actually isn't
    ('Cambridge Bay', 'CBB', 1),
    ('Kugluktu', 'KUG', 2),
    ('Resolute', 'RES', 1),
    ('Iqaluit', 'IQA', 1),
    ('Ottawa', 'OTT', 2),
    ('Pond Inlet', 'PON', 2),
    ('Churchill', 'CHU', 2),
    ('Coral Harbour', 'COR', 2),
    ('Fort McMurray', 'MCM', 3), #
    ('City of Dawson', 'DAW', 3), #
    ('Ministik Lake', 'EDM', 3), #
    ('Grise Fiord', 'GRI', 2),
    ('Gjoa Haven', 'GJO', 3), #
    ('Fort Simpson', 'FSI', 3), #
    ('Qikiqtarjuaq', 'QIK', 3), #
    ('Gilliam', 'GIL', 3), #
    ('Fort Smith', 'FSM', 3), #
    ('Rabbit Lake', 'RAB', 3), #
    ('Rankin Inlet', 'RAN', 3), #
    ('Taloyoak/Vernadsky Station', 'TAL', 3), #
    ('Whale Cove', 'WHA', 2),
    ('Uapishka', 'UAP', 2),
    ('Repulse Bay', 'REP', 2),
    ('Sanikiluaq', 'SAN', 2),
    ('Sachs Harbour', 'SAC', 3),
]

def select_station():
    def on_select(abbr):
        nonlocal EUR
        EUR = abbr
        root.destroy()

    def on_enter(e):
        if e.widget['state'] == 'normal':
            e.widget.config(bg='#d1e7dd', font=('Arial', 14))

    def on_leave(e):
        if e.widget['state'] == 'normal':
            e.widget.config(bg='SystemButtonFace', font=('Arial', 14))

    def on_press(e):
        if e.widget['state'] == 'normal':
            e.widget.config(bg='#a3c9a8')

    def on_release(e):
        if e.widget['state'] == 'normal':
            e.widget.config(bg='#d1e7dd')

    root = tk.Tk()
    root.title('Select Station')
    root.iconbitmap('fuip3101.ico')
    EUR = None

    label = tk.Label(root, text="Select station:", font=('Arial', 14))
    label.pack(padx=10, pady=10)

    btn_frame = tk.Frame(root)
    btn_frame.pack(padx=10, pady=10)

    # 4 columns
    columns = 4
    for idx, (city, abbr, number) in enumerate(eur_options):
        state = 'normal' if number == 4 else 'disabled'
        btn = tk.Button(
            btn_frame,
            text=city,
            width=24,
            height=2,
            font=('Arial', 14),
            relief='raised',
            bd=2,
            command=lambda a=abbr: on_select(a),
            state=state,
        )
        if number != 4:
            btn.config(bg="#969494", fg="#000000")
        row = idx // columns
        col = idx % columns
        btn.grid(row=row, column=col, padx=5, pady=5, sticky='ew')
        btn.bind('<Enter>', on_enter)
        btn.bind('<Leave>', on_leave)
        btn.bind('<ButtonPress-1>', on_press)
        btn.bind('<ButtonRelease-1>', on_release)

    root.mainloop()
    return EUR

EUR = select_station()

if EUR is None or EUR.strip() == '':
    print('No station selected for EUR variable.')
    messagebox.showerror('Error', 'No station selected 😦💔🥀.')
    exit()





import time

# pretend loading bar
progress_root = tk.Tk()
progress_root.title('Processing sTEC from '+EUR)
progress_root.iconbitmap('fuip3101.ico')  # cat
progress_label = tk.Label(progress_root, text='Calculations in progress...\nThis may take a while... ⏳🫷', padx=30, pady=20, font=('Arial',30))
progress_label.pack()

progress_bar = tk.Canvas(progress_root, width=300, height=22, bg='white', bd=2, relief='sunken')
progress_bar.pack(pady=10)
bar = progress_bar.create_rectangle(0, 0, 0, 22, fill='green')

progress_root.update()

# pretend loading bar animation
colors = ['#a3e4d7', '#aed6f1', '#d2b4de', '#f9e79f', '#f5b7b1', '#fad7a0', '#abebc6']
steps = 27
import itertools
spinner = itertools.cycle(['🕐', '🕑', '🕒', '🕓', '🕔', '🕕', '🕖', '🕗', '🕘', '🕙', '🕚', '🕛'])
for i in range(steps):
    progress_label.config(text=f'Calculations in progress...\nThis may take a while... ⏳🫷\n\t{next(spinner)}')
    color = colors[i % len(colors)]
    progress_bar.itemconfig(bar, fill=color)
    progress_bar.coords(bar, 0, 0, 10*(i+1), 22)
    progress_root.update()
    time.sleep(0.10)





if(HH>=10):
    url_rinex=('https://www.rspl.ca/data/gnss/data/highrate/'+str(YYYY)+'/'+str(DDD)+'/'+str(YY)+'d/'+str(HH)+'/'+EUR+'C00CAN_R_'+str(YYYY)+str(DDD)+str(HH)+'00_15M_01S_GO.crx.gz')
else:
    url_rinex=('https://www.rspl.ca/data/gnss/data/highrate/'+str(YYYY)+'/'+str(DDD)+'/'+str(YY)+'d/0'+str(HH)+'/'+EUR+'C00CAN_R_'+str(YYYY)+str(DDD)+'0'+str(HH)+'00_15M_01S_GO.crx.gz')

def download_file(url):
    local_filename = url.split('/')[-1]
    with requests.get(url, stream=True, verify=False) as r:
        with open(local_filename, 'wb') as f:
            shutil.copyfileobj(r.raw, f)

    return local_filename
    print(local_filename)
print(url_rinex)
download_file(url_rinex)

if(HH>=10):
    compressed=EUR+'C00CAN_R_'+str(YYYY)+str(DDD)+str(HH)+'00_15M_01S_GO.crx.gz'
    uncompressed=EUR+'C00CAN_R_'+str(YYYY)+str(DDD)+str(HH)+'00_15M_01S_GO.rnx'
    hatanaka.decompress_on_disk(compressed)
    print('decompressed saved at ' + uncompressed)
else:
    compressed=EUR+'C00CAN_R_'+str(YYYY)+str(DDD)+'0'+str(HH)+'00_15M_01S_GO.crx.gz'
    uncompressed=EUR+'C00CAN_R_'+str(YYYY)+str(DDD)+'0'+str(HH)+'00_15M_01S_GO.rnx'
    hatanaka.decompress_on_disk(compressed)
    print('decompressed saved at ' + uncompressed)

rinex_data = gr.load(uncompressed)

f1 = 1575420000 #dr1=0.16237244751199471499166472281867
f2 = 1227600000 #dr2=0.26741840036072685144043892821996
#the 0.104m/TECU in the doc is 0.10504595284873213644877420540129m/TECU I guess
r=0.10504595284873213644877420540129



P1 = rinex_data['C1C']
P2 = rinex_data['C2L']



TECp= (P2-P1)/r #pseudorange tec

C1= rinex_data['L1C'] #CP is *not* a great abbreviation for things...
C2= rinex_data['L2W']

c=299792458
w1=c/f1
w2=c/f2

print('w1, w2:',w1,w2)



os.remove(compressed)
os.remove(uncompressed)
#w1=0.1905
#w2=0.2445

#w1=190293.67279836486
#w2=244210.21342456827

TECc=((w1*C1)-(w2*C2))/(r) #carrier phase tec
#TECc=((c/C1)-(c/C2))/r

times = rinex_data.coords['time'].values
sats = rinex_data.coords['sv'].values

time_grid, sat_grid = np.meshgrid(times, sats, indexing='ij')


#TECp_flat = TECp.values.flatten()
TECc_flat = TECc.values.flatten()
time_flat = time_grid.flatten()
sat_flat = sat_grid.flatten()



#TECp = TECp.values
TECc = TECc.values

#valid_indices = np.isfinite(TECp) & np.isfinite(TECc)
#TECp = TECp[valid_indices]
#TECc = TECc[valid_indices]


'''
def rquare(offset, TECc, TECp):
    return TECp + (TECc + offset)

TECo, _ = leastsq(rquare,[100], args=(TECc, TECp))

TEC=(TECc-TECo)
'''

'''
TEC_df = pd.DataFrame({'TEC': TEC})
TEC_df.to_csv('TEC_values_'+EUR+str(YYYY)+str(DDD)+'0'+str(HH)+'.csv', index=False)

print('TEC values calculated and saved to TEC_values_'+str(YYYY)+str(DDD)+'0'+str(HH)+'.csv.')
print('TECo', TECo) ###
print('TECp', TECp) ###
print('TECc', TECc) ###
print('TEC', TEC) ###




print('Final TEC values or smthn without bias fixing: ' + str(TEC))
'''





# thank you david themens 💕


lam1=0.1905
lam2=0.244
npk1=0 # integer phase ambiguity in L1
npk2=0 # integer phase ambiguity in L2
A=40.3


npk= (lam1*npk1)-(lam2*npk2)
'''
IpkfP= -iIpkfL = A*sTECpk/f^2
sTECpk = IpkfP*(f^2)/A
'''
fmhz1 = 1575.420000 #dr1=0.16237244751199471499166472281867
fmhz2 = 1227.600000
fghz1=fmhz1/1000
fghz2=fmhz2/1000
dp1=0
dp2=0
dk1=0
dk2=0
phip1=0
phip2=0
phik1=0
phik2=0
DCBp=dp1-dp2
DCBk=dk1-dk2
DPBp=phip1-phip2
DPBk=phik1-phik2
PPKGF=P2-P1
LPKGF=(w1*C1)-(w2*C2)
# PsTECpk=((PPKGF+(c*(DPBp+DPBk)))/A)*(((f1*f1)-(f2*f2))/(f1*f1*f2*f2))
LsTECpk=((LPKGF-npk+(c*(DPBp+DPBk)))/A)*(((f1*f1)-(f2*f2))/(f1*f1*f2*f2))
DCBpk=DCBp+DCBk
W=0
#  sTECpk=(1/A)*((f1*f1*f2*f2)/((f1*f1)-(f2*f2)))*(LPKGF+W+(c*DCBpk))
# Aff=(((1/A))*((fmhz1*fmhz1*fmhz2*fmhz2)/((fmhz1*fmhz1)-(fmhz2*fmhz2))))
Aff=(((1/A)*((f1*f1*f2*f2)/((f1*f1)-(f2*f2))))*(1/10000000000000000))
sTECpk=Aff*(LPKGF+W+(c*DCBpk))
psTECpk=Aff*(PPKGF+W+(c*DCBpk))
print('LPKGF:',LPKGF)
print('ghd',Aff)
print('sTECPK:',sTECpk)
print('LsTECpk:',LsTECpk)
#print('PsTECpk:',PsTECpk)
print('sTECpk:',sTECpk)
print('psTECpk:',psTECpk)

dpsTECpk=np.diff(psTECpk)
dsTECpk=np.diff(sTECpk)

print('dsTECpk:',dsTECpk)
          
LPKGF_series = (w1*C1.values)-(w2*C2.values)
sTECpk_series = (Aff * (LPKGF_series + W + (c*DCBpk)))
dsTECpk_series = np.diff(sTECpk_series)

PPKGF_series = P2.values-P1.values
psTECpk_series= (Aff * (PPKGF_series + W + (c*DCBpk)))
dpsTECpk_series = np.diff(psTECpk_series)





print('psTECpk:',psTECpk)
print('dpsTECpk:',dpsTECpk)



'''
valid_indices = np.isfinite(TECp_flat) & np.isfinite(TECc_flat)
TECp_flat = TECp_flat[valid_indices]
TECc_flat = TECc_flat[valid_indices]
time_flat = time_flat[valid_indices]
sat_flat = sat_flat[valid_indices]
'''


df = pd.DataFrame({
    'time': pd.to_datetime(time_flat),
    'satellite thingymabob': sat_flat,
    #'TECp': TECp_flat,
    'TECc': TECc_flat
})

df.to_csv('TEC_values_'+str(YYYY)+str(DDD)+'0'+str(HH)+'.csv', index=False)


'''
# Individual plots for each satellite
for sat in df['satellite'].unique():
    sat_df = df[df['satellite'] == sat]
    plt.figure(figsize=(10,5))
    plt.plot(sat_df['time'], sat_df['TECp'], label='TECp (code)', marker='o', linestyle='-')
    plt.plot(sat_df['time'], sat_df['TECc'], label='TECc (phase)', marker='x', linestyle='--')
    plt.title(f'TEC for {sat}')
    plt.xlabel('Time')
    plt.ylabel('TEC (TECU)')
    plt.legend()
    plt.tight_layout()
    plt.show()



# Combined plot for all satellites
plt.figure(figsize=(12, 6))
for sat in df['satellite thingymabob'].unique():
    sat_df = df[df['satellite thingymabob'] == sat]
    plt.plot(sat_df['time'], sat_df['TECp'], label=f'TECp {sat}', linestyle='-')
    plt.plot(sat_df['time'], sat_df['TECc'], label=f'TECc {sat}', linestyle='--')
plt.title('TEC for each satellite thingymabob')
plt.xlabel('Time')
plt.ylabel('TEC (TECU)')
plt.legend()
plt.tight_layout()
plt.show()
'''

print('TEC values calculated, saved, and plotted.')



progress_label.config(text='calculated,\n press "OKay 🙂" to view plots,\ndon\'t mind the cat.')
progress_bar.coords(bar, 0, 0, 320, 22)
progress_root.update()

ok_button = tk.Button(progress_root, text='OKay 🙂', command=progress_root.destroy, padx=20, pady=10)
ok_button.pack(pady=10)
progress_root.mainloop()



# tigger

from PIL import Image, ImageTk
import random

def show_tigger():
    tigger_photos = [
        "fuip3101.png",
        "yo89ndz8.png",
        "9e366f7i.png",
        "stanley.png",
    ]

    tigger = tk.Tk()
    tigger.title("meow")
    tigger.iconbitmap('fuip3101.ico')
    img_path= random.choice(tigger_photos)
    img = Image.open(img_path)
    #img = img.resize((300, 300))
    tigger_img = ImageTk.PhotoImage(img)
    img_label = tk.Label(tigger, image=tigger_img)
    img_label.pack(padx=10, pady=10)
    msg = tk.Label(tigger, text="meow", font=("Arial", 16))
    msg.pack(pady=10)
    def close_and_continue():
        tigger.destroy()
    ok_btn = tk.Button(tigger, text="meow", font=("Arial", 14), command=close_and_continue, padx=20, pady=10)
    ok_btn.pack(pady=10)
    tigger.mainloop()

show_tigger()






import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, Button




# toggleable lines
fig, ax = plt.subplots(figsize=(12, 6))
lines = []
labels = []



df['sTECpk'] = sTECpk_series.flatten()
#df['dsTECpk_series'] = dsTECpk_series.flatten()


df['dsTECpk'] = np.nan  # initialize with NaN




for sat in df['satellite thingymabob'].unique():
    sat_mask = df['satellite thingymabob'] == sat
    sat_df = df[sat_mask].sort_values('time')
    # Calculate the difference and pad with NaN to keep length
    ds = np.diff(sat_df['sTECpk'], prepend=np.nan)
    df.loc[sat_mask, 'dsTECpk'] = ds


for sat in df['satellite thingymabob'].unique():
    sat_df = df[df['satellite thingymabob'] == sat]
    #l1, = ax.plot(sat_df['time'], sat_df['TECp'], label=f'TECp {sat} (code)', linestyle='-')
    #l2, = ax.plot(sat_df['time'], sat_df['TECc'], label=f'TECc {sat} (phase)', linestyle='--')

    l3, = ax.plot(sat_df['time'], sat_df['sTECpk'], label=f'sTECpk {sat}', linestyle='-.')
    l4, = ax.plot(sat_df['time'], sat_df['dsTECpk'], label=f'dsTECpk {sat}', linestyle=':')
    lines.extend([l3, l4]) # l1, l2
    labels.extend([f'sTECpk {sat}', f'dsTECpk {sat}']) # f'TECp {sat} (code)', f'TECc {sat} (phase)', 


plt.title('TEC (phase) for each satellite thingymabob ('+EUR+')')
plt.xlabel('Time')
plt.ylabel('TEC (TECU)')
plt.tight_layout()

# check buttons
rax = plt.axes([0.01, 0.15, 0.15, 0.75])  
check = CheckButtons(rax, labels, [True]*len(labels))

def line_togle(label):
    index = labels.index(label)
    lines[index].set_visible(not lines[index].get_visible())
    plt.draw()

check.on_clicked(line_togle)

# toggle all button
toggle_ax = plt.axes([0.01, 0.90, 0.15, 0.05])
toggle_button = Button(toggle_ax, 'Toggle all (click twice)')
toggle_state = [True] 

def toggle_all(event):
    new_state = not all(line.get_visible() for line in lines)
    for i, line in enumerate(lines):
        line.set_visible(new_state)
        check.set_active(i) if check.get_status()[i] != new_state else None
    plt.draw()
    toggle_state[0] = new_state

toggle_button.on_clicked(toggle_all)

plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.subplots_adjust(left=0.22) 
plt.show()








fig2, ax2 = plt.subplots(figsize=(12, 6))
lines2 = []
labels2 = []




# Add psTECpk and dpsTECpk to the DataFrame
df['psTECpk'] = psTECpk_series.flatten()
df['dpsTECpk'] = np.nan




for sat in df['satellite thingymabob'].unique():
    sat_mask = df['satellite thingymabob'] == sat
    sat_df = df[sat_mask].sort_values('time')
    ds = np.diff(sat_df['psTECpk'], prepend=np.nan)
    df.loc[sat_mask, 'dpsTECpk'] = ds

for sat in df['satellite thingymabob'].unique():
    sat_df = df[df['satellite thingymabob'] == sat]
    l1, = ax2.plot(sat_df['time'], sat_df['psTECpk'], label=f'psTECpk {sat}', linestyle='-.')
    l2, = ax2.plot(sat_df['time'], sat_df['dpsTECpk'], label=f'dpsTECpk {sat}', linestyle=':')
    lines2.extend([l1, l2])
    labels2.extend([f'psTECpk {sat}', f'dpsTECpk {sat}'])

ax2.set_title('TEC (code) for each satellite thingymabob ('+EUR+')')
ax2.set_xlabel('Time')
ax2.set_ylabel('psTECpk (units)')
plt.tight_layout()

# Check buttons for second plot
rax2 = plt.axes([0.01, 0.15, 0.15, 0.75], figure=fig2)
check2 = CheckButtons(rax2, labels2, [True]*len(labels2))

def line_toggle2(label):
    index = labels2.index(label)
    lines2[index].set_visible(not lines2[index].get_visible())
    plt.draw()

check2.on_clicked(line_toggle2)

# Toggle all button for second plot
toggle_ax2 = plt.axes([0.01, 0.90, 0.15, 0.05], figure=fig2)
toggle_button2 = Button(toggle_ax2, 'Toggle all (click twice)')
toggle_state2 = [True]

def toggle_all2(event):
    new_state = not all(line.get_visible() for line in lines2)
    for i, line in enumerate(lines2):
        line.set_visible(new_state)
        check2.set_active(i) if check2.get_status()[i] != new_state else None
    plt.draw()
    toggle_state2[0] = new_state

toggle_button2.on_clicked(toggle_all2)

ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.subplots_adjust(left=0.22)
plt.show()



'''
plot_root = tk.Tk()
plot_root.title("TEC Plots")


notebook = ttk.Notebook(plot_root)
notebook.pack(fill='both', expand=True)

# sTECpk and dsTECpk
fig1, ax1 = plt.subplots(figsize=(12, 6))
lines1 = []
labels1 = []

for sat in df['satellite thingymabob'].unique():
    sat_df = df[df['satellite thingymabob'] == sat]
    l3, = ax1.plot(sat_df['time'], sat_df['sTECpk'], label=f'sTECpk {sat}', linestyle='-.')
    l4, = ax1.plot(sat_df['time'], sat_df['dsTECpk'], label=f'dsTECpk {sat}', linestyle=':')
    lines1.extend([l3, l4])
    labels1.extend([f'sTECpk {sat}', f'dsTECpk {sat}'])

ax1.set_title('TEC (phase) for each satellite thingymabob ('+EUR+')')
ax1.set_xlabel('Time')
ax1.set_ylabel('TEC (TECU)')
ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
fig1.tight_layout()

tab1 = ttk.Frame(notebook)
notebook.add(tab1, text="Phase TEC")

canvas1 = FigureCanvasTkAgg(fig1, master=tab1)
canvas1.draw()
canvas1.get_tk_widget().pack(fill='both', expand=True)

# psTECpk and dpsTECpk
fig2, ax2 = plt.subplots(figsize=(12, 6))
lines2 = []
labels2 = []

for sat in df['satellite thingymabob'].unique():
    sat_df = df[df['satellite thingymabob'] == sat]
    l1, = ax2.plot(sat_df['time'], sat_df['psTECpk'], label=f'psTECpk {sat}', linestyle='-.')
    l2, = ax2.plot(sat_df['time'], sat_df['dpsTECpk'], label=f'dpsTECpk {sat}', linestyle=':')
    lines2.extend([l1, l2])
    labels2.extend([f'psTECpk {sat}', f'dpsTECpk {sat}'])

ax2.set_title('TEC (code) for each satellite thingymabob ('+EUR+')')
ax2.set_xlabel('Time')
ax2.set_ylabel('psTECpk (units)')
ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
fig2.tight_layout()

tab2 = ttk.Frame(notebook)
notebook.add(tab2, text="Code TEC")

canvas2 = FigureCanvasTkAgg(fig2, master=tab2)
canvas2.draw()
canvas2.get_tk_widget().pack(fill='both', expand=True)



def legend_interactive(fig, ax, lines):
    leg = ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    lined = {}
    for legline, origline in zip(leg.get_lines(), lines):
        legline.set_picker(True)
        lined[legline] = origline

    def on_pick(event):
        legline = event.artist
        origline = lined[legline]
        vis = not origline.get_visible()
        origline.set_visible(vis)
        legline.set_alpha(1.0 if vis else 0.2)
        fig.canvas.draw()

    fig.canvas.mpl_connect('pick_event', on_pick)


legend_interactive(fig1, ax1, lines1)
legend_interactive(fig2, ax2, lines2)
'''

'''
def add_boxes(tab, lines, labels, canvas):
   
    control_frame = tk.Frame(tab)
    control_frame.pack(side='right', fill='y', padx=10, pady=10)
    vars = []
    for i, label in enumerate(labels):
        var = tk.BooleanVar(value=True)
        cb = tk.Checkbutton(control_frame, text=label, variable=var)
        cb.pack(anchor='w')
        vars.append(var)
        def toggle_line(idx=i):
            lines[idx].set_visible(vars[idx].get())
            canvas.draw()
        var.trace_add('write', lambda *args, idx=i: toggle_line(idx))
    # Toggle all button
    def toggle_all():
        new_state = not all(v.get() for v in vars)
        for v in vars:
            v.set(new_state)
    btn = tk.Button(control_frame, text="Toggle all")
    btn.pack(pady=10)
    btn.config(command=toggle_all)

add_boxes(tab1, lines1, labels1, canvas1)
add_boxes(tab2, lines2, labels2, canvas2)
'''



'''
def fake_checkbox_legend(fig, ax, lines, labels):
    checked = [True] * len(lines)
    def get_label(i):
        return f"{'☑' if checked[i] else '☐'} {labels[i]}"

    leg = ax.legend([get_label(i) for i in range(len(labels))], loc='upper left', bbox_to_anchor=(1, 1))
    lined = {}
    for i, (legline, origline) in enumerate(zip(leg.get_lines(), lines)):
        legline.set_picker(True)
        lined[legline] = (origline, i)
    def on_pick(event):
        legline = event.artist
        origline, idx = lined[legline]
        checked[idx] = not checked[idx]
        origline.set_visible(checked[idx])
        leg.set_texts([get_label(i) for i in range(len(labels))])
        legline.set_alpha(1.0 if checked[idx] else 0.2)
        fig.canvas.draw()
    fig.canvas.mpl_connect('pick_event', on_pick)
    from matplotlib.widgets import Button
    toggle_ax = fig.add_axes([0.81, 0.05, 0.15, 0.05])
    toggle_button = Button(toggle_ax, 'Toggle all')
    def toggle_all(event):
        new_state = not all(checked)
        for idx, (line, legline) in enumerate(zip(lines, leg.get_lines())):
            checked[idx] = new_state
            line.set_visible(new_state)
            legline.set_alpha(1.0 if new_state else 0.2)
        leg.set_texts([get_label(i) for i in range(len(labels))])
        fig.canvas.draw()
    toggle_button.on_clicked(toggle_all)

fake_checkbox_legend(fig1, ax1, lines1, labels1)
fake_checkbox_legend(fig2, ax2, lines2, labels2)
'''



#plot_root.mainloop()
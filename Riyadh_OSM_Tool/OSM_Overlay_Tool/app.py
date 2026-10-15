"""Standalone desktop UI. No changes to the road-texture application."""
import csv
import json
import math
import os
import queue
import threading
import subprocess
from roads_style import prepare_job
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import ImageTk
import satellite as SAT
from aligned_pair import export_pair
from core import VERSION, Cancelled, build_index, cache_path, cache_valid, dimensions, extent, export_bundle, query_roads, render


class App:
    def __init__(self, root):
        self.root=root
        root.title('OSM + Satellite Tool '+VERSION)
        root.geometry(f'{min(1200,root.winfo_screenwidth()-40)}x{min(810,root.winfo_screenheight()-100)}');root.minsize(1000,620)
        root.configure(bg='#142330')
        self.events=queue.Queue();self.cancel=threading.Event();self.busy=False
        self.batch_ids=[];self.index=None;self.result=None;self.photo=None
        self.cache_dir=Path(os.environ.get('LOCALAPPDATA',str(Path.home()/'.cache')))/'OSMOverlayTool'/'cache'
        self.source=tk.StringVar();self.status=tk.StringVar(value='Choose a PBF, then build its road index.')
        self.coords={k:tk.StringVar(value=v) for k,v in [('left','46.67'),('right','46.68'),('top','24.72'),('bottom','24.71')]}
        self.width=tk.StringVar(value='4096');self.height=tk.StringVar(value='4096')
        self.auto=tk.BooleanVar(value=True);self.other=tk.BooleanVar(value=False)
        self.line=tk.StringVar(value='4');self.zoom=tk.StringVar(value='20')
        self.name=tk.StringVar(value='riyadh_overlay');self.tile=tk.StringVar(value='')
        self.standard_street=None;self.street_path=None;self.satellite=None;self.mode=tk.StringVar(value='OSM');self.opacity=tk.DoubleVar(value=0.8)
        self.umd_exe=tk.StringVar();self.map_type=tk.StringVar(value='Choose map source')
        self.map_id=tk.StringVar();self.manual_crop=tk.BooleanVar(value=False)
        self.download_dir=tk.StringVar(value=str(Path.home()/'Documents'/'OSM_Satellite_Downloads'))
        self.preferences=self.cache_dir.parent/'satellite_settings.json'
        try:
            saved=json.loads(self.preferences.read_text())
            self.umd_exe.set(saved.get('downloader',''));self.map_id.set(saved.get('map_id',''))
            self.map_type.set(saved.get('map_type','Choose map source'));self.download_dir.set(saved.get('directory',self.download_dir.get()))
        except (OSError,ValueError):pass
        self.grid={}
        grid=Path(__file__).with_name('riyadh_grid.csv')
        if grid.exists():
            with grid.open(newline='',encoding='utf8') as stream:
                self.grid={r['id']:r for r in csv.DictReader(stream)}
        extra=Path(__file__).with_name('riyadh_500m_grid.csv')
        if extra.exists():
            with extra.open(newline='',encoding='utf8') as stream:self.grid.update({r['id']:r for r in csv.DictReader(stream)})
        style=ttk.Style();style.theme_use('clam')
        style.configure('TFrame',background='#f4f7fa')
        style.configure('TLabel',background='#f4f7fa',foreground='#142330',font=('Segoe UI',10))
        style.configure('TButton',font=('Segoe UI',10),padding=7)
        style.configure('TCheckbutton',background='#f4f7fa',font=('Segoe UI',10))
        style.configure('Title.TLabel',font=('Segoe UI',17,'bold'))
        style.configure('Hint.TLabel',font=('Segoe UI',9),foreground='#526579')
        body=ttk.Frame(root,padding=14);body.pack(fill='both',expand=True)
        panel=ttk.Frame(body,width=350);panel.pack(side='left',fill='y',padx=(0,14));panel.pack_propagate(False)
        scroll=tk.Canvas(panel,bg='#f4f7fa',highlightthickness=0,width=330)
        scrollbar=ttk.Scrollbar(panel,orient='vertical',command=scroll.yview)
        scrollbar.pack(side='right',fill='y');scroll.pack(side='left',fill='both',expand=True)
        scroll.configure(yscrollcommand=scrollbar.set)
        controls=ttk.Frame(scroll);window=scroll.create_window((0,0),window=controls,anchor='nw')
        controls.bind('<Configure>',lambda event:scroll.configure(scrollregion=scroll.bbox('all')))
        scroll.bind('<Configure>',lambda event:scroll.itemconfigure(window,width=event.width))
        def wheel(event):
            widget=root.winfo_containing(event.x_root,event.y_root)
            if widget and str(widget).startswith(str(panel)):scroll.yview_scroll(-int(event.delta/120),'units')
        root.bind_all('<MouseWheel>',wheel)
        ttk.Label(controls,text='OSM + satellite',style='Title.TLabel').pack(anchor='w')
        ttk.Label(controls,text='Read-only PBF input | Local disk cache',style='Hint.TLabel').pack(anchor='w',pady=(2,12))
        ttk.Label(controls,text='1. Source file').pack(anchor='w')
        ttk.Entry(controls,textvariable=self.source,state='readonly').pack(fill='x',pady=5)
        ttk.Button(controls,text='Browse PBF / OSM...',command=self.choose).pack(fill='x')
        self.index_button=ttk.Button(controls,text='Build / load road index',command=self.index_source)
        self.index_button.pack(fill='x',pady=(5,12))
        ttk.Label(controls,text='2. Coordinates (WGS 84 decimal degrees)').pack(anchor='w')
        form=ttk.Frame(controls);form.pack(fill='x',pady=5)
        labels={'left':'Left longitude','right':'Right longitude','top':'Top latitude','bottom':'Bottom latitude'}
        for i,(key,var) in enumerate(self.coords.items()):
            ttk.Label(form,text=labels[key]).grid(row=i,column=0,sticky='w',pady=3)
            ttk.Entry(form,textvariable=var,width=19).grid(row=i,column=1,sticky='ew',padx=(8,0),pady=3)
        if self.grid:
            row=ttk.Frame(controls);row.pack(fill='x',pady=5)
            ttk.Entry(row,textvariable=self.tile,width=15).pack(side='left')
            ttk.Button(row,text='Load grid ID',command=self.load_tile).pack(side='left',padx=5)
            ttk.Label(controls,text='Optional: RYD_0001 ... RYD_4581 from the PDF',style='Hint.TLabel').pack(anchor='w')
        ttk.Checkbutton(controls,text='Also include paths and other highway types',variable=self.other).pack(anchor='w',pady=8)
        ttk.Label(controls,text='3. Output size and line style').pack(anchor='w',pady=(5,0))
        row=ttk.Frame(controls);row.pack(fill='x',pady=5)
        ttk.Label(row,text='Width').pack(side='left');ttk.Entry(row,textvariable=self.width,width=8).pack(side='left',padx=5)
        ttk.Label(row,text='Height').pack(side='left');self.height_entry=ttk.Entry(row,textvariable=self.height,width=8)
        self.height_entry.pack(side='left',padx=5)
        ttk.Checkbutton(controls,text='Calculate height from geographic aspect',variable=self.auto,command=self.aspect).pack(anchor='w')
        row=ttk.Frame(controls);row.pack(fill='x',pady=5)
        ttk.Label(row,text='Zoom guide').pack(side='left')
        ttk.Combobox(row,textvariable=self.zoom,values=['18','19','20','21'],width=4,state='readonly').pack(side='left',padx=5)
        ttk.Button(row,text='Size from zoom',command=self.size_from_zoom).pack(side='left')
        ttk.Label(controls,text='Zoom guide assumes 256 px Web Mercator tiles.',style='Hint.TLabel').pack(anchor='w')
        row=ttk.Frame(controls);row.pack(fill='x',pady=6)
        ttk.Label(row,text='Export line width (px)').pack(side='left')
        ttk.Spinbox(row,from_=1,to=30,textvariable=self.line,width=6).pack(side='left',padx=7)
        self.preview_button=ttk.Button(controls,text='Preview selected roads',command=self.preview)
        self.preview_button.pack(fill='x',pady=(5,8))
        ttk.Label(controls,text='4. Export name').pack(anchor='w')
        ttk.Entry(controls,textvariable=self.name).pack(fill='x',pady=5)
        self.export_button=ttk.Button(controls,text='Export overlay bundle...',command=self.export)
        self.export_button.pack(fill='x')
        self.cancel_button=ttk.Button(controls,text='Cancel current operation',command=self.cancel.set,state='disabled')
        self.cancel_button.pack(fill='x',pady=5)
        ttk.Separator(controls).pack(fill='x',pady=12)
        ttk.Label(controls,text='5. Universal Maps Downloader',style='Title.TLabel',wraplength=320).pack(anchor='w')
        ttk.Entry(controls,textvariable=self.umd_exe,state='readonly').pack(fill='x',pady=5)
        ttk.Button(controls,text='Choose downloader.exe...',command=self.choose_downloader).pack(fill='x')
        selector=ttk.Combobox(controls,textvariable=self.map_type,values=[*SAT.MAP_TYPES,'Custom map type ID'],state='readonly')
        selector.pack(fill='x',pady=6);selector.bind('<<ComboboxSelected>>',self.select_map)
        row=ttk.Frame(controls);row.pack(fill='x',pady=4)
        ttk.Label(row,text='Map type ID').pack(side='left');ttk.Entry(row,textvariable=self.map_id,width=9).pack(side='left',padx=6)
        ttk.Label(controls,text='Uses the coordinate fields and zoom above.',style='Hint.TLabel').pack(anchor='w')
        ttk.Entry(controls,textvariable=self.download_dir,state='readonly').pack(fill='x',pady=5)
        ttk.Button(controls,text='Choose map download folder...',command=self.choose_download_dir).pack(fill='x')
        self.download_button=ttk.Button(controls,text='Download selected map + preview',command=self.download_satellite)
        self.download_button.pack(fill='x',pady=6)
        self.load_sat_button=ttk.Button(controls,text='Load existing satellite image...',command=self.load_satellite)
        self.load_sat_button.pack(fill='x')
        self.english_button=ttk.Button(controls,text='Download English street map',command=lambda:self.download_street('Google Street Map (English)'))
        self.english_button.pack(fill='x',pady=5)
        self.no_labels_button=ttk.Button(controls,text='Download street map without labels',command=lambda:self.download_street('Google Street Map (without labels)'))
        self.no_labels_button.pack(fill='x')
        ttk.Button(controls,text='Load / clear roads-only map...',command=self.load_roads).pack(fill='x')
        self.pair_button=ttk.Button(controls,text='Export aligned set...',command=self.export_aligned_pair)
        self.pair_button.pack(fill='x',pady=6)
        self.batch_import_button=ttk.Button(controls,text='Import batch IDs.txt...',command=self.import_batch)
        self.batch_import_button.pack(fill='x',pady=5)
        self.batch_button=ttk.Button(controls,text='Start / resume batch',command=self.start_batch)
        self.batch_button.pack(fill='x')
        ttk.Label(controls,text='Native satellite crop + matching OSM overlay. Export size fields apply only to overlay-only export.',style='Hint.TLabel',wraplength=310).pack(anchor='w')
        ttk.Checkbutton(controls,text='Existing image is an exact EPSG:3857 crop\nof the entered bounds (manual fallback)',variable=self.manual_crop).pack(anchor='w',pady=6)
        ttk.Label(controls,text='Georeferenced TIFF preferred. Downloader installation and provider access are required.',style='Hint.TLabel',wraplength=310).pack(anchor='w',pady=(0,10))
        right=ttk.Frame(body);right.pack(side='left',fill='both',expand=True)
        ttk.Label(right,text='Preview | OSM grid: north up, EPSG:3857',style='Title.TLabel').pack(anchor='w')
        ttk.Label(right,text='Cyan = OSM paths. These lines do not represent asphalt width.',style='Hint.TLabel').pack(anchor='w',pady=(4,10))
        toolbar=ttk.Frame(right);toolbar.pack(fill='x',pady=(0,6))
        for name in ('OSM','Satellite','Overlay','Street map'):
            ttk.Radiobutton(toolbar,text=name,variable=self.mode,value=name,command=self.paint).pack(side='left',padx=(0,8))
        ttk.Label(toolbar,text='Line opacity').pack(side='left',padx=(10,4))
        ttk.Scale(toolbar,from_=0,to=1,variable=self.opacity,command=lambda value:self.paint(),length=120).pack(side='left')
        self.canvas=tk.Canvas(right,bg='#142330',highlightthickness=0);self.canvas.pack(fill='both',expand=True)
        self.canvas.bind('<Configure>',lambda event:self.paint())
        self.stats=ttk.Label(right,text='Load a source and preview an area.',wraplength=690)
        self.stats.pack(anchor='w',pady=8)
        ttk.Label(right,text='To match satellite imagery, use its ACTUAL geographic bounds, projection and pixel size. A downloader can return whole tiles beyond your requested bounds. Grid IDs use the PDF download envelope, not the final 1 km UTM crop.',style='Hint.TLabel',wraplength=690).pack(anchor='w',pady=(0,5))
        bottom=ttk.Frame(root,padding=(14,8));bottom.pack(fill='x')
        self.bar=ttk.Progressbar(bottom,mode='indeterminate');self.bar.pack(side='right',fill='x',padx=8)
        ttk.Label(bottom,textvariable=self.status,wraplength=900).pack(side='left',anchor='w')
        self.aspect();root.after(100,self.poll);root.protocol('WM_DELETE_WINDOW',self.close)

    def close(self):
        self.cancel.set();self.root.destroy()

    def select_map(self,event=None):
        if self.map_type.get() in SAT.MAP_TYPES:self.map_id.set(str(SAT.MAP_TYPES[self.map_type.get()]))
        else:self.map_id.set('')

    def save_preferences(self):
        self.preferences.parent.mkdir(parents=True,exist_ok=True)
        self.preferences.write_text(json.dumps(dict(downloader=self.umd_exe.get(),map_id=self.map_id.get(),
                    map_type=self.map_type.get(),directory=self.download_dir.get()),indent=2),encoding='utf8')

    def choose_downloader(self):
        if self.busy:return
        path=filedialog.askopenfilename(title='Select Universal Maps Downloader\\downloader.exe',filetypes=[('Downloader','downloader.exe'),('Executable','*.exe')])
        if path:self.umd_exe.set(path)

    def choose_download_dir(self):
        path=filedialog.askdirectory(title='Choose map download folder')
        if path:self.download_dir.set(path)

    def sat_result(self,path,settings,index,manual=False,folder=None):
        sat=SAT.load_image(path,settings['bounds'] if manual else None)
        roads=query_roads(index,settings['bounds'],settings['other']) if index else []
        return dict(kind='satellite',satellite=sat,settings=settings,roads=roads,download_folder=folder)

    def import_batch(self):
        if self.busy:return
        path=filedialog.askopenfilename(title='One RYD grid ID per line',filetypes=[('Text files','*.txt')])
        if not path:return
        try:
            from batch import read_ids
            self.batch_ids=read_ids(path,self.grid)
            self.status.set(f'{len(self.batch_ids)} unique IDs queued. Batch uses Google satellite and street map without labels.')
        except (ValueError,OSError) as error:messagebox.showerror('Batch IDs',str(error))

    def start_batch(self):
        if self.busy:return
        try:
            if not self.batch_ids:raise ValueError('Import a batch IDs.txt first.')
            if not self.index:raise ValueError('Build or load the PBF road index first.')
            zoom=int(self.zoom.get());line=int(self.line.get())
            if not 1<=line<=30:raise ValueError('Line width must be 1-30.')
            exe=self.umd_exe.get();index=self.index;ids=list(self.batch_ids);minor=self.other.get()
            SAT.commands(exe,self.download_dir.get(),'batch_check',1,zoom,tuple(float(self.grid[ids[0]][k]) for k in ('left','right','top','bottom')))
        except (ValueError,OSError) as error:messagebox.showerror('Batch settings',str(error));return
        parent=filedialog.askdirectory(title='Batch output folder (choose same folder to resume)')
        if not parent:return
        def work():
            from batch import run
            summary=run(ids,self.grid,exe,parent,index,zoom,line,minor,progress=lambda text:self.events.put(('status',text)),cancelled=self.cancel.is_set)
            return dict(kind='batch',summary=summary,folder=parent)
        self.start(work)

    def download_street(self,label):
        if self.busy:return
        self.map_type.set(label);self.select_map();self.download_satellite()

    def download_satellite(self):
        if self.busy:return
        try:
            settings=self.settings();zoom=int(self.zoom.get());map_id=int(self.map_id.get())
            exe=self.umd_exe.get();parent=self.download_dir.get();name=self.name.get().strip()
            if not parent:raise ValueError('Choose a satellite download folder.')
            SAT.commands(exe,parent,name,map_id,zoom,settings['bounds'])
            if SAT.tile_estimate(settings['bounds'],zoom)>1600:raise ValueError('Use a smaller area or lower zoom: this selection exceeds 1,600 standard tiles.')
            self.save_preferences();index=self.index;street=map_id in SAT.STREET_TYPES.values()
        except (ValueError,OSError) as error:messagebox.showerror('Map download settings',str(error));return
        def work():
            path,folder=SAT.download(exe,parent,name+'_street' if street else name,map_id,zoom,settings['bounds'],
                  progress=lambda text:self.events.put(('status',text)),cancelled=self.cancel.is_set)
            if self.cancel.is_set():raise Cancelled('Cancelled before preview.')
            result=self.sat_result(path,settings,index,folder=folder)
            if street:result['kind']='street'
            return result
        self.start(work)

    def load_satellite(self):
        if self.busy:return
        try:settings=self.settings()
        except ValueError as error:messagebox.showerror('Coordinates',str(error));return
        path=filedialog.askopenfilename(title='Load combined satellite image',filetypes=[('Satellite images','*.tif *.tiff *.png *.jpg *.jpeg'),('All files','*.*')])
        if path:
            manual=self.manual_crop.get();index=self.index
            self.start(lambda:self.sat_result(path,settings,index,manual))

    def download_roads(self):
        if self.busy:return
        try:
            settings=self.settings();exe=Path(self.umd_exe.get())
            if not exe.is_file() or exe.name.lower()!='downloader.exe':raise ValueError('Choose Universal Maps Downloader downloader.exe first.')
            folder,text=prepare_job(self.download_dir.get(),self.name.get().strip(),settings['bounds'],int(self.zoom.get()))
            self.root.clipboard_clear();self.root.clipboard_append(text);self.root.update()
            subprocess.Popen([str(exe)],shell=False)
            os.startfile(str(folder))
            messagebox.showinfo('Roads-only download','Style copied to clipboard. In UMD choose Google Maps (Customized Stylers and Languages), Custom Parameters > normal > Import, and paste. Use DOWNLOAD_INSTRUCTIONS.txt in the opened folder for bounds and zoom. After combining to georeferenced TIFF, use Load roads-only map here.')
        except (ValueError,OSError) as error:messagebox.showerror('Roads-only map',str(error))

    def load_roads(self):
        if self.busy:return
        path=filedialog.askopenfilename(title='Load roads-only map (Cancel keeps current map)',filetypes=[('Georeferenced maps','*.tif *.tiff *.png')])
        if path:
            try:
                data=SAT.load_image(path)
                if not data['geo']:raise ValueError('Geographic metadata is required; use TIFF in metres from Map Combiner.')
                self.street_path=path;self.status.set('Roads-only map loaded: '+Path(path).name+'; it will be included in the aligned set.')
            except (ValueError,OSError) as error:messagebox.showerror('Road map',str(error))
        elif self.street_path and messagebox.askyesno('Road map','Clear the currently loaded roads-only map?'):
            self.street_path=None;self.status.set('Road map cleared.')

    def aspect(self):
        self.height_entry.configure(state='disabled' if self.auto.get() else 'normal')

    def choose(self):
        if self.busy:return
        path=filedialog.askopenfilename(title='Select OSM source',filetypes=[('OSM data','*.pbf *.osm'),('All files','*.*')])
        if path:
            self.source.set(path);self.index=None;self.result=None;self.paint()
            self.status.set('Source selected. Build / load its index next.')

    def load_tile(self):
        try:
            key=self.tile.get().strip().upper();row=self.grid[key]
            for field in self.coords:self.coords[field].set(row[field])
            self.name.set(key);self.status.set('Loaded '+key+' download envelope. Final 1 km cropping is a later step.')
        except KeyError:messagebox.showerror('Grid ID','Enter a valid ID such as RYD_0001.')

    def bounds(self):
        from core import validate_bounds
        try:return validate_bounds(*(float(self.coords[k].get()) for k in ('left','right','top','bottom')))
        except ValueError as error:raise ValueError('Check coordinate fields: '+str(error))

    def settings(self):
        bounds=self.bounds();width=int(self.width.get());height=None if self.auto.get() else int(self.height.get())
        width,height=dimensions(bounds,width,height);line=int(self.line.get())
        if not 1<=line<=30:raise ValueError('Line width must be 1-30 pixels.')
        self.height.set(str(height))
        return dict(bounds=bounds,width=width,height=height,line=line,other=self.other.get())

    def size_from_zoom(self):
        try:
            ex=extent(self.bounds());step=2*math.pi*6378137/(256*2**int(self.zoom.get()))
            width=max(64,round((ex[2]-ex[0])/step));height=max(64,round((ex[3]-ex[1])/step))
            dimensions(self.bounds(),width,height)
            self.width.set(str(width));self.height.set(str(height));self.auto.set(False);self.aspect()
            self.status.set('Size calculated for the exact coordinate envelope; actual downloaded tile bounds may be larger.')
        except ValueError as error:messagebox.showerror('Size',str(error))

    def start(self,work):
        if self.busy:return
        self.busy=True;self.cancel.clear()
        for button in (self.index_button,self.preview_button,self.export_button,self.download_button,self.load_sat_button,self.pair_button,self.english_button,self.no_labels_button,self.batch_import_button,self.batch_button):button.configure(state='disabled')
        self.cancel_button.configure(state='normal');self.bar.start(12)
        def run():
            try:self.events.put(('done',work()))
            except Exception as error:self.events.put(('error',str(error)))
        threading.Thread(target=run,daemon=True).start()

    def poll(self):
        try:
            while True:
                kind,payload=self.events.get_nowait()
                if kind=='status':self.status.set(payload);continue
                self.busy=False;self.bar.stop();self.cancel_button.configure(state='disabled')
                for button in (self.index_button,self.preview_button,self.export_button,self.download_button,self.load_sat_button,self.pair_button,self.english_button,self.no_labels_button,self.batch_import_button,self.batch_button):button.configure(state='normal')
                if kind=='error':
                    self.status.set(payload);messagebox.showerror('Operation stopped',payload)
                elif payload['kind']=='index':
                    self.index=payload['index'];self.status.set('Road index ready. Enter coordinates and preview.')
                elif payload['kind']=='batch':
                    summary=payload['summary']
                    text=f"Batch: {len(summary['completed'])} completed, {len(summary['skipped'])} skipped, {len(summary['failed'])} failed."
                    if summary['cancelled']:text+=' Stopped; resume with the same IDs and output folder.'
                    self.status.set(text);messagebox.showinfo('Batch results',text+'\nSaved in: '+str(payload['folder'])+'\nSee batch_summary.json for failures.')
                elif payload['kind']=='pair':
                    self.status.set('Aligned pair saved: '+str(payload['folder']))
                    mpp=payload['metadata']['mask_tool_metres_per_pixel']
                    scale=(f'{mpp:.8f}' if mpp is not None else 'Non-square pixels: reproject before width estimation')
                    messagebox.showinfo('Aligned pair exported',f"Saved in:\n{payload['folder']}\n\nUse satellite.png and overlay.png in Road Mask Tool. Downloaded street maps are saved as street_map_downloaded.png; a custom roads-only map adds street_roads.png.\nGround metres/pixel: {scale}\nInspect comparison.png for OSM offsets.")
                elif payload['kind']=='street':
                    self.standard_street=payload['satellite'];self.mode.set('Street map');self.paint()
                    self.status.set('Street map downloaded and combined: '+str(payload['download_folder']))
                    self.stats.configure(text='Street map saved separately; satellite remains the reference for aligned exports.')
                elif payload['kind']=='satellite':
                    self.satellite=payload['satellite'];self.result=payload;self.mode.set('Satellite');self.paint()
                    geo=self.satellite['geo']
                    note=('Geographic metadata loaded; Overlay view is available.' if geo else self.satellite['note'])
                    if geo and not payload['roads']:note+=' Load the PBF index and preview roads to add lines.'
                    if geo and geo.get('manual'):note='Bounds manually assigned by you; verify the crop before using Overlay.'
                    self.stats.configure(text=f"Satellite: {Path(self.satellite['path']).name} | {note}")
                    self.status.set('Satellite preview ready. '+note)
                else:
                    self.result=payload;self.paint()
                    settings=payload['settings'];self.stats.configure(text=f"{len(payload['roads']):,} road ways | Export: {settings['width']:,} x {settings['height']:,} px | Preview is reduced to fit")
                    if payload.get('folder'):
                        self.status.set('Export saved: '+str(payload['folder']))
                        messagebox.showinfo('Export complete','Saved in:\n'+str(payload['folder'])+'\n\nKeep overlay.png, overlay.pgw and overlay.prj together.')
                    else:self.status.set('Preview ready. Export uses the current coordinate and size fields.')
        except queue.Empty:pass
        self.root.after(100,self.poll)

    def index_source(self):
        source=self.source.get()
        if not source:messagebox.showerror('Source','Choose your PBF first.');return
        self.start(lambda:dict(kind='index',index=build_index(source,self.cache_dir,
                   progress=lambda text:self.events.put(('status',text)),cancelled=self.cancel.is_set)))

    def prepare(self):
        settings=self.settings()
        if not self.index:raise ValueError('Build / load the PBF road index first.')
        path,signature=cache_path(self.source.get(),self.cache_dir)
        if path!=self.index or not cache_valid(path,signature):raise ValueError('The source changed. Build / load the index again.')
        return settings,self.index

    def generate(self,settings,index,folder=None,name=None):
        self.events.put(('status','Extracting and clipping roads to your coordinates...'))
        roads=query_roads(index,settings['bounds'],settings['other'])
        if self.cancel.is_set():raise Cancelled('Cancelled before rendering.')
        result=dict(kind='preview',roads=roads,settings=settings)
        if folder:
            self.events.put(('status','Writing transparent overlay and geographic metadata...'))
            result['folder']=export_bundle(folder,name,roads,settings['bounds'],settings['width'],settings['height'],settings['line'],index)
        return result

    def preview(self):
        try:settings,index=self.prepare()
        except (ValueError,OSError) as error:messagebox.showerror('Input',str(error));return
        self.start(lambda:self.generate(settings,index))

    def export(self):
        try:settings,index=self.prepare()
        except (ValueError,OSError) as error:messagebox.showerror('Input',str(error));return
        folder=filedialog.askdirectory(title='Choose destination for a new export folder')
        if folder:
            name=self.name.get().strip();self.start(lambda:self.generate(settings,index,folder,name))

    def export_aligned_pair(self):
        try:
            settings,index=self.prepare()
            if not self.satellite or not self.satellite.get('geo'):
                raise ValueError('Download or load a georeferenced satellite image first.')
            sat=self.satellite;street_path=self.street_path;standard_path=self.standard_street['path'] if self.standard_street else None
            name=self.name.get().strip()
        except (ValueError,OSError) as error:
            messagebox.showerror('Aligned pair',str(error));return
        folder=filedialog.askdirectory(title='Choose destination for aligned pair')
        if not folder:return
        def work():
            output,metadata=export_pair(folder,name,sat,settings['bounds'],index,settings['line'],settings['other'],
                                      self.cancel.is_set,lambda text:self.events.put(('status',text)),street_path=street_path,standard_path=standard_path)
            return dict(kind='pair',folder=output,metadata=metadata)
        self.start(work)

    def paint(self):
        self.canvas.delete('all')
        cw,ch=max(1,self.canvas.winfo_width()),max(1,self.canvas.winfo_height())
        if self.mode.get()=='Street map':
            if not self.standard_street:
                self.canvas.create_text(cw/2,ch/2,text='Select Google Street Map and click Download selected map.',fill='#aac1d2');return
            im=self.standard_street['image'].copy();im.thumbnail((max(1,cw-24),max(1,ch-24)))
            self.photo=ImageTk.PhotoImage(im);self.canvas.create_image(cw/2,ch/2,image=self.photo);return
        if self.mode.get()!='OSM' and self.satellite:
            sat=self.satellite
            if not sat['geo']:
                if self.mode.get()=='Overlay':
                    self.canvas.create_text(cw/2,ch/2,text='Overlay disabled: no supported geographic metadata.\nLoad a georeferenced export or an explicitly verified exact crop.',fill='#aac1d2',font=('Segoe UI',12),width=cw-50);return
                im=sat['image'].copy();im.thumbnail((max(1,cw-24),max(1,ch-24)))
            else:
                settings=self.result['settings'] if self.result else self.settings()
                w,h=settings['width'],settings['height'];scale=min((cw-24)/w,(ch-24)/h,1)
                pw,ph=max(64,round(w*scale)),max(64,round(h*scale))
                im=SAT.preview_on_grid(sat,settings['bounds'],(pw,ph)).convert('RGBA')
                if self.mode.get()=='Overlay' and self.result:
                    overlay=render(self.result['roads'],settings['bounds'],pw,ph,max(1,round(settings['line']*scale)),True)
                    alpha=overlay.getchannel('A').point(lambda value:round(value*self.opacity.get()))
                    overlay.putalpha(alpha);im.alpha_composite(overlay)
            self.photo=ImageTk.PhotoImage(im);self.canvas.create_image(cw/2,ch/2,image=self.photo);return
        if self.mode.get()!='OSM' and not self.satellite:
            self.canvas.create_text(cw/2,ch/2,text='Download or load a satellite image first.',fill='#aac1d2',font=('Segoe UI',14));return
        if not self.result:
            self.canvas.create_text(cw/2,ch/2,text='OSM road preview',fill='#aac1d2',font=('Segoe UI',16));return
        settings=self.result['settings'];w,h=settings['width'],settings['height']
        scale=min((cw-24)/w,(ch-24)/h,1)
        pw,ph=max(64,round(w*scale)),max(64,round(h*scale))
        im=render(self.result['roads'],settings['bounds'],pw,ph,max(1,round(settings['line']*scale)),False)
        self.photo=ImageTk.PhotoImage(im)
        self.canvas.create_image(cw/2,ch/2,image=self.photo)


if __name__=='__main__':
    root=tk.Tk();App(root);root.mainloop()

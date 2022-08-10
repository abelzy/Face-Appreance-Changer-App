
import os
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
from Config_File import getConfig
from matplotlib import pyplot
import array as arr
import PIL.ImageTk
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk,Image 

class App(Frame):
    
    def __init__(self, master):
        Frame.__init__(self, master)
        self.img_domain = ["Orginal Image","Black Hair","Blond Hair","Brown Hair","Sex","Age"]
        self.num_page=0
        self.img_type = StringVar()
        self.img_type.set(self.img_domain[self.num_page])
        self.result_dir =  "./stargan_celeba_256/results/"
        self.la = Label(self)
        self.la.pack()
        self.pack()
        img_path = os.path.join(self.result_dir, '{}-images.jpg'.format(self.num_page))
        self.im =  PIL.Image.open(img_path)
        self.chg_image()
        
    def gen_image(self):
        cudnn.benchmark = False
        config = getConfig();
        celeba_loader = get_loader(config.celeba_image_dir, config.attr_path, config.selected_attrs,
                                   config.celeba_crop_size, config.image_size, config.batch_size,
                                   'CelebA', config.mode, config.num_workers)
        solver = Solver(celeba_loader, None, config)
        solver.test()
        
    def chg_image(self):
        
        if self.im.mode == "1": # bitmap image
            self.img = PIL.ImageTk.BitmapImage(self.im, foreground="white")
        else:              # photo image
            self.img = PIL.ImageTk.PhotoImage(self.im)
        self.la.config(image=self.img, bg="#000000",
                       width=self.img.width(), height=self.img.height())
                       


    def open(self):
        filename = filedialog.askopenfilename()
        if filename != "":
            print(filename)
            self.im = PIL.Image.open(filename)
            self.im_name= "./data/000001.jpg"
            self.im.save(self.im_name)
            self.im =PIL.Image.open("./data/000001.jpg")
            self.gen_image()
            img_path = os.path.join(self.result_dir, '{}-images.jpg'.format(self.num_page))
            self.im =  PIL.Image.open(img_path)
            self.chg_image()
            self.img_type.set(self.img_domain[self.num_page])
        

    def seek_next(self):
        if(self.num_page <=4):
            self.num_page+=1
            img_path = os.path.join(self.result_dir, '{}-images.jpg'.format(self.num_page))
            self.im =  PIL.Image.open(img_path)
            self.chg_image()
            self.img_type.set(self.img_domain[self.num_page])
        
    def seek_prev(self):
        if(self.num_page >=1):
            self.num_page-=1
            img_path = os.path.join(self.result_dir, '{}-images.jpg'.format(self.num_page))
            self.im =  PIL.Image.open(img_path)
            self.chg_image()
            self.img_type.set(self.img_domain[self.num_page])


if __name__ == "__main__":
    root = Tk()
    root.title("STARGAN")
    root.minsize(width=600,height=500)
    root.maxsize(width=600,height=500)
    root.wm_attributes("-transparent", "grey")
    root.geometry("600x500")
    



    # Adding a background image
    background_image =Image.open("lib.jpg")
    [imageSizeWidth, imageSizeHeight] = background_image.size      
    img = ImageTk.PhotoImage(background_image)
    Canvas1 = Canvas(root)
    Canvas1.create_image(300,340,image = img)      
    Canvas1.config(bg="white",width = imageSizeWidth, height = imageSizeHeight)
    Canvas1.pack(expand=True,fill=BOTH)
    
    appFrame = Frame(root,bg='black')
    appFrame.place(relx=0.28,rely=0.35,)
    app = App(appFrame)
    
    Frame1 = Frame(root,bd=5)
    Frame1.place(relx=0.02,rely=0.06,relwidth=0.25,relheight=0.10)
    open_Btn =Button(Frame1, text="Upload New Image",bg='black', bd = 5,fg='white',font=('Courier',10), command=app.open)
    open_Btn.place(relx=0,rely=0,relwidth=1,relheight=1)
    
    # Creating a photoimage object to use image
    Frame2 = Frame(root,bd=5)
    Frame2.place(relx=0.1,rely=0.53,relwidth=0.15,relheight=0.15)
    prev_icon = PhotoImage(file = r"prev.png") 
    prev_Btn =Button(Frame2,image = prev_icon,width=70,height=60,relief=FLAT,command=app.seek_prev)
    prev_Btn.place(relx=0,rely=0,relwidth=1,relheight=1) 
    
    Frame3 = Frame(root,bd=5)
    Frame3.place(relx=0.74,rely=0.53,relwidth=0.15,relheight=0.15)
    next_icon = PhotoImage(file = r"next.png") 
    next_Btn =Button(Frame3,image = next_icon,width=70,height=60,relief=FLAT,command=app.seek_next)
    next_Btn.place(relx=0,rely=0,relwidth=1,relheight=1)
    
    Frame4 = Frame(root,bg="white",bd=5)
    Frame4.place(relx=0.35,rely=0.22,relwidth=0.3,relheight=0.11)
    img_txt = Label(Frame4, textvariable=app.img_type,bg="white",font=("Courier ",13))
    img_txt.place(relx=0,rely=0,relwidth=1,relheight=1)
    
    
      
    root.mainloop()

    
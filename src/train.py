import torch
import torch.nn as nn
import torch.optim as optim
from preprocessing import preprocess
from model import VAE, vae_loss

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(train_loader,valid_loader,model,optimizer,num_epochs):

    for epoch in range(num_epochs):
        model.train()
        for i,(images,_) in enumerate(train_loader):
            images=images.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(images)
            loss=vae_loss(recon,images,mu,logvar)
            loss.backward()
            optimizer.step()
            
        print(f"Epoch [{epoch+1}/{num_epochs}], train_Loss: {loss.item()}")
            

        model.eval()
        with torch.no_grad():
            for i,(images,_) in enumerate(valid_loader):
                images=images.to(device)
                recon, mu, logvar = model(images)
                loss=vae_loss(recon,images,mu,logvar)
            print(f"Epoch [{epoch+1}/{num_epochs}], val_Loss: {loss.item()}")
    
    torch.save(model.state_dict(),"models/model.pth")
            


def main():
    #defining the parameters
    num_epochs=30
    lr=0.001
    batch_size=64
    latent_dim=512

    model=VAE(latent_dim=latent_dim).to(device)
    optimizer=optim.Adam(model.parameters(),lr=lr)

    #preprocessing the data
    train_loader,valid_loader=preprocess()

    #training the model
    train_model(train_loader,valid_loader,model,optimizer,num_epochs)


if __name__=="__main__":
    main()
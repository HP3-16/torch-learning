import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
class MovieDataset:
    def __init__(self, users, movies, ratings):
        self.users = users
        self.movies = movies
        self.ratings = ratings
    
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, item):
        user = self.users[item]
        movie = self.movies[item]
        rating = self.ratings[item]

        return {"users":torch.tensor(user,dtype=torch.long),
                "movies":torch.tensor(movie,dtype=torch.long),
                "ratings":torch.tensor(rating,dtype=torch.float)
                }


#Model
class Recommender(nn.Module):
    def __init__(self, n_users, n_movies):
        super().__init__()
        self.user_embed = nn.Embedding(n_users,embedding_dim=32)
        self.movie_embed = nn.Embedding(n_movies,embedding_dim=32)
        self.out = nn.Linear(64,1)

    def RMSE_(self, output,ratings):
        output = output.detach().cpu().numpy()
        ratings = ratings.detach().cpu().numpy()
        return {
            "rmse":np.sqrt(mean_squared_error(ratings,output))
        }
    
    def forward(self, users, movies, ratings):
        user_embeds = self.user_embed(users)
        movie_embeds = self.movie_embed(movies)
        ouptut = torch.cat([user_embeds,movie_embeds],dim=1)
        output = self.out(output)
    
        loss = nn.MSELoss()(output,ratings.view(-1,1))  
        metrics = self.RMSE_(output,ratings.view(-1,1))
        return output,loss,metrics
        
    



def train():
    df = pd.read_csv("../input/train_v2.csv")
    user_label = LabelEncoder()
    movie_label = LabelEncoder()

    df.user = user_label.fit_transform(df.user.values)
    df.movie = movie_label.fit_transform(df.movie.values)
    df_train, df_val  = train_test_split(df,test_size=0.2,random_state=37,stratify=df.rating.values)
    train_dataset  = MovieDataset(users = df_train.user.values, movies= df_train.movie.values, ratings=df_train.rating.values)
    val_dataset  = MovieDataset(users = df_val.user.values, movies= df_val.movie.values, ratings=df_val.rating.values)


if __name__ == "main":
    train()
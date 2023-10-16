import pandas
import torchtext
from torchtext.data import Field,Dataset,Example
# reference taken from the original source code @ https://pytorch.org/text/_modules/torchtext/data/example.html#Example.fromlist

class SeriesExample(Example):
    @classmethod
    def fromdict(cls,data,fields):
        ex = cls()
        for key,field in fields.items():
            if field is not None:
                setattr(ex,key,field.preprocess(data[key]))
            else:
                setattr(ex, key, data[key])
        return ex
            

    @classmethod
    def fromSeries(cls,data,fields):
        return cls.fromdict(data.to_dict(),fields)
    


class DatasetFromDataFrame(Dataset):
    def __init__(self,examples,fields,filter_pred=None):
        self.examples = examples.apply(SeriesExample.fromSeries,args=(fields,),axis=1).tolist()
        if filter_pred is not None:
            self.examples = filter(filter_pred, self.examples)
            self.fields = dict(fields)
             
            for n, f in list(self.fields.items()):
                 if isinstance(n, tuple):
                     self.fields.update(zip(n, f))
                     del self.fields[n]  
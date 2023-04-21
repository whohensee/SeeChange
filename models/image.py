
from models.base import Base


class Image(Base):
    __tablename__ = 'images'


if __name__ == '__main__':
    im = Image()
    print(im.id)

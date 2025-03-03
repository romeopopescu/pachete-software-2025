from typing import Annotated
from fastapi import FastAPI, Depends, Query, HTTPException
from sqlmodel import Field, Session, SQLModel, create_engine, select

class Student (SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    email: str = Field(index=True)
    age: int | None = Field(default=None, index=True)


sqlite_file_name = "database.sqlite"
sqlite_url = f"sqlite:///{sqlite_file_name}"

connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, connect_args=connect_args)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session

SessionDep = Annotated[Session, Depends(get_session)]

app = FastAPI()

@app.on_event("startup")
def on_startup():
    create_db_and_tables()

@app.post("/students/")
def create_student(student: Student, session: SessionDep) -> Student:
    session.add(student)
    session.commit()
    session.refresh(student)
    return student

@app.get("/students/")
def read_students(
    session: SessionDep,
    offset: int = 0,
    limit: Annotated[int, Query(le=100)] = 100,
    ) -> list[Student]:
    students = session.exec(select(Student).offset(offset).limit(limit)).all()
    return students


# @app.get("/")
# async def root():
#     return {"message": "Hello world"}

# @app.get("/foo")
# async def root():
#     return {"message": "fool"}

# @app.get("/items/{item_id}")
# async def root(item_id: int):
#     return {"item_id": item_id}

# @app.post("/")
# async def root():
#     return {"message": "POST method"}
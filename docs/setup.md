## Setting up a SeeChange instance

### Installing using Docker

At the moment, some of the things below will not work if you install Docker Desktop.  
It has to do with permissions and bind-mounting system volumes; 
because of how Docker Desktop works, the files inside the container all end up owned as root, 
not as you, even if they are owned by you on your own filesystem.  Hopefully there's a way to fix this, 
but in the meantime, install Docker Engine instead of Docker Desktop; instructions are here:

- Installing Docker Engine : https://docs.docker.com/engine/install/
- Setting up rootless mode (so you don't have to sudo everything) : https://docs.docker.com/engine/security/rootless/

#### Development shell -- local database

To create a database on your local machine and get a development shell in which to run code, 
cd into the `devshell` directory and run
```
   USERID=<UID> GROUPID=<GID> docker compose up -d
```
replacing `<UID>` with your uid, and `<GID>` with your GID.  
You can avoid typing this all the time by creating a file `.env` in the `devshell` directory with contents
```
  USERID=<UID>
  GROUPID=<GID>
```
again replacing `<UID>` and `<GID>` with the right things (you can find your uid and gid with `id -u` and `id -g`). 

Once things are started, there will be two containers running, 
one for your database, one to which you can attach with a shell.  Do
```
   docker ps
```
to see what the created containers are named; the shell container is usually `devshell-seechange-1`.  
You can then get a shell inside that environment with
```
   docker exec -it devshell-seechange-1 /bin/bash
```
and you're in.  (Put in the right name for the container if it's not devshell-seechange-1.)

This docker image bind-mounts your seechange checkout 
(the parent directory of the `devshell` directory where you're working) at `/seechange`.  
That means if you work in that directory, it's the same as working in the checkout.  
If you edit something outside the container, 
the differences will be immediately available inside the container (since it's the same physical filesystem).  
This means there's no need to rebuild the container every time you change any bit of code.

When you're done, exit the container, and run
```
  USERID=<UID> GROUPID=<GID> docker compose down
```
to stop and delete the container images.  (If you created the `.env` file mentioned above, 
you don't need the USERID and GROUPID definitions, and can just type `docker compose down`.)

The `docker-compose.yaml` file in this directory defines a volume where postgres stores its data.  
This means that every time you restart the environment, 
the database will still be as it was before.  
This isn't what you want for running tests, but it's often what you want for development.  
You can see what volumes are defined with
```
  docker volume list
```
In that list, you should see something that has a name `devshell_seechange-postgres-dbdata`.  
If you want to wipe this volume out and start with a fresh database, you can run
```
  docker volume rm devshell_seechange-postgres-dbdata
```


#### Development shell -- using an external existing database

TBD


#### Running tests

To run the tests on your local system in an environment that approximates how they'll be run on github, 
cd into `tests` and run the following command (which requires the "docker compose CLI plugin" installed to work):
```
   export GITHUB_REPOSITORY_OWNER=<yourname>
   docker compose build
   USERID=<uid> GROUIP=<gid> docker compose run runtests
```
where you replace `<uid>` and `<gid>` with your own userid and groupid; if you don't do this, the tests will run, 
but various pycache files will get created in your checkout owned by root, which is annoying. 
At the end, `echo $?`; if 0, that's a pass, if 1 (or anything else not 0), that's a fail.  
(The output you see to the screen should tell you the same information.)  
This will take a long time the first time you do it, as it has to build the docker images, 
but after that, it should be fast (unless the Dockerfile has changed for either image).  
The variable GITHUB_RESPOSITORY_OWNER must be set to *something*; it only matters if you try to push or pull the images.  
Try setting it to your github username, though if you really want to push and pull you're going to have to look up 
making tokens on github.  (The docker-compose.yaml file is written to run on github, which is why it includes this variable.)

After the test is complete, run
```
    docker compose down
```
(otherwise, the postgres container will still be running).


### Database migrations

Database migrations are handled with alembic.

If you've just created a database and want to initialize it with all the tables, run
```
  alembic upgrade head
```

After editing any schema, you have to create new database migrations to apply them.  Do this by running something like:
```
  alembic revision --autogenerate -m "<put a short comment here>"
```
The comment will go in the filename, so it should really be short.  
Look out for any warnings, and review the created migration file before applying it (with `alembic upgrade head`).


### Installing SeeChange on a local machine (not dockerized)

As always, checkout the code from github: <https://github.com/c3-time-domain/SeeChange>.
We recommend using a virtual environment to install the dependencies. For example:

```bash
python3 -m venv venv
```

Then activate the virtual environment and install the dependencies:

```bash
cd SeeChange
source venv/bin/activate
pip install -r requirements.txt
```

This covers the basic python dependencies. 

Install some of the standalone executables needed for 
analyzing astronomical images:

```bash
sudo apt install source-extractor psfex scamp swarp
sudo ln -sf /usr/bin/python3 /usr/bin/python
sudo ln -sf /usr/bin/SWarp /usr/bin/swarp
```

The last two lines will create links to the executables
with more commonly used spellings. 

Now we need to install postgres and set it up 
to run on the default port (5432) with a database called `seechange`.

On a mac, you can do this with homebrew:

```bash
brew install postgresql
brew services start postgresql
/usr/local/opt/postgres/bin/createdb seechange
```

Usually you will want to add the default user:
    
```bash
/usr/local/opt/postgres/bin/createuser -s postgres
```

On linux/debian use
 
```bash
sudo apt install postgresql
```

Make sure the default port is 5432 in /etc/postgresql/14/main/postgresql.conf
(assuming the version of postgres is 14).
To restart the service do 

```bash
sudo service postgresql restart
```


To log in to postgres (as the user "postgres"): 
```bash
sudo -u postgres psql
```

From here you can create or drop the database:

```sql
CREATE DATABASE seechange;
DROP DATABASE seechange WITH(force);
```

To use the database, login as above but then change into the database:

```sql
\c seechange
```


#### Installing Q3C extension for postgres

Get the code from <https://github.com/segasai/q3c>. 
Installing following the instructions: 

```bash
make
make install
```

Login to psql and do:

```sql
\c seechange
CREATE EXTENSION q3c;
```

#### Getting the database schema up-to-date

The database schema is managed by alembic.
If the database is in a fresh state (just created), do:

```bash
alembic upgrade head
```

If the database has already been used (e.g., on a different branch), you may need to do:

```bash
alembic downgrade base
alembic upgrade head
```

To generate a new migration script, do:

```bash
alembic revision --autogenerate -m "message"
```

#### Installing submodules

To install submodules used by SeeChange 
(e.g., the `nersc-upload-connector` package that is used to connect the archive)
do the following:

```bash
cd extern/nersc-upload-connector
git submodule init
```

If those packages require updates, you can do that from the root SeeChange directory
using:

```bash
git submodule update
```

Note that you can just do 

```bash
git submodule update --init
```

from the root directory, which will also initialize any 
submodules that have not been initialized yet.

#### Setting up environmental variables

Some environmental variables are used by SeeChange.
 - `GITHUB_REPOSITORY_OWNER` is the name of your github user (used only for dockerized tests). 
 - `SEECHANGE_TEST_ARCHIVE_DIR` is used to set up a local directory for test data archive. 
Usually this will point to a folder outside the SeeChange directory, 
where data can be downloaded and stored.
 - `SEECHANGE_CONFIG` can be used to specify the location of the main config file,
but if that is not defined, SeeChange will just use the default config at the top level 
of the SeeChange directory, or the one in the `tests` directory (when running local tests). 

#### Adding local config files for tests

To allow tests to find the archive and the local database, 
a custom config file needs to be loaded. 
The default file, in `SeeChange/tests/seechange_config_test.yaml`,
will automatically look for (and load) two local config files named
`local_overrides.yaml` and `local_augments.yaml`. 
The first will override any keys in the default config, 
and the second one will update the existing parameter dictionaries and lists. 

One way to set things up is to put the following into 
`SeeChange/tests/local_augments.yaml`:

```yaml
archive:
  local_read_dir: /path/to/local/archive
  archive_url: null

db:
  engine: postgresql
  user: postgres
  password: fragile
  host: localhost
  port: 5432
  database: seechange
```

Replace `/path/to/local/archive` with the path to the local archive directory, 
which should also be defined as the environmental variable `SEECHANGE_TEST_ARCHIVE_DIR`.

The same files (`local_overrides.yaml` and `local_augments.yaml`) can be used
on the main SeeChange directory, where they have the same effect, 
just for running a real instance of the SeeChange pipeline locally. 

#### Running the tests

At this point the tests should be working from the IDE or from the command line:

```bash
pytest --ignore=extern
```

The extern folder includes submodules that do not support local testing at this point. 

You can also add a `.pytest.ini` file with the following: 

```
[pytest]
testpaths =
    tests
```

Which will limit pytest to automatically only run the tests in that folder, 
and ignore other test folders, e.g., those in the `extern` folder.

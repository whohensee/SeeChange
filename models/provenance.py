import json
import git
import hashlib
import sqlalchemy as sa
from sqlalchemy import event
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB

from models.base import engine, Base, SmartSession


class CodeHash(Base):
    __tablename__ = "code_hashes"

    def __init__(self, git_hash):
        self.hash = git_hash

    hash = sa.Column(sa.String, index=True, unique=True)

    code_version_id = sa.Column(sa.Integer, sa.ForeignKey("code_versions.id", ondelete="CASCADE"))

    code_version = relationship("CodeVersion", back_populates="code_hashes")


class CodeVersion(Base):
    __tablename__ = 'code_versions'

    version = sa.Column(
        sa.String,
        nullable=False,
        index=True,
        unique=True,
        doc='Version of the code. Can use semantic versioning or date/time, etc. '
    )

    code_hashes = sa.orm.relationship(
        CodeHash,
        backref='code_versions',
        cascade='all, delete-orphan',
        passive_deletes=True,
        doc='List of commit hashes for this version of the code',
    )

    def update(self, session=None):
        repo = git.Repo(search_parent_directories=True)
        git_hash = repo.head.object.hexsha

        with SmartSession(session) as session:
            hash_obj = session.scalars(sa.select(CodeHash).where(CodeHash.hash == git_hash)).first()
            if hash_obj is None:
                hash_obj = CodeHash(git_hash)

            self.code_hashes.append(hash_obj)


class Provenance(Base):
    __tablename__ = "provenances"

    process = sa.Column(
        sa.String,
        nullable=False,
        index=True,
        doc="Name of the process (pipe line step) that produced these results. "
    )

    code_version_id = sa.Column(
        sa.ForeignKey("code_versions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="ID of the code version the provenance is associated with. ",
    )

    code_version = relationship(
        "CodeVersion",
        back_populates="provenances",
        cascade="save-update, merge, expunge, refresh-expire",
        lazy='selectin',
    )

    parameters = sa.Column(
        JSONB,
        nullable=False,
        default={},
        doc="Critical parameters used to generate the underlying data. ",
    )

    upstream_ids = sa.Column(
        sa.ARRAY(sa.Integer),
        nullable=False,
        default=[],
        doc="IDs of the provenance rows used in upstream analysis. ",
    )

    unique_hash = sa.Column(
        sa.String,
        nullable=False,
        index=True,
        unique=True,
        doc="Unique hash of the code version, parameters and upstream provenances used to generate this dataset. ",
    )

    def update_hash(self):
        """
        Update the unique_hash using the code_version, parameters and upstream_ids.
        """
        if self.process is None or self.parameters is None or self.upstream_ids is None or self.code_version is None:
            raise ValueError('Provenance must have process, code_version, parameters and upstream_ids defined. ')

        superdict = dict(
            process=self.process,
            parameters=self.parameters,
            upstream_ids=self.upstream_ids,
            code_version=self.code_version.version
        )
        json_string = json.dumps(superdict, sort_keys=True)
        self.unique_hash = hashlib.sha256(json_string.encode("utf-8")).hexdigest()


CodeVersion.provenances = relationship(
    "Provenance",
    back_populates="code_version",
    cascade="save-update, merge, expunge, refresh-expire",
    foreign_keys="Provenance.code_version_id",
    passive_deletes=True,
)


CodeHash.metadata.create_all(engine)
CodeVersion.metadata.create_all(engine)
Provenance.metadata.create_all(engine)


@event.listens_for(Provenance, "before_insert")
def insert_new_dataset(mapper, connection, target):
    """
    This function is called before a new provenance is inserted into the database.
    It will check all the required fields are populated and update the unique_hash.
    """
    # TODO: check to see that all upstream_ids actually exist
    bad_ids = []
    for i in target.upstream_ids:
        if i is None:
            bad_ids.append(i)
        if not isinstance(i, int):
            bad_ids.append(i)
        upstream = connection.execute(sa.select(Provenance).where(Provenance.id == i)).first()
        if upstream is None:
            bad_ids.append(i)
    if len(bad_ids) > 0:
        raise ValueError(
            f'The upstream_ids {bad_ids} are not integers or refer to non-existent Provenance rows. '
        )
    target.update_hash()



if __name__ == "__main__":
    pass

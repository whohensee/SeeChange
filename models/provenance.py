import json
import git
import hashlib
import sqlalchemy as sa
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB

from models.base import engine, Base


class CodeVersion(Base):
    __tablename__ = 'code_versions'

    version = sa.Column(
        sa.String,
        nullable=False,
        index=True,
        unique=True,
        doc='Version of the code. Can use semantic versioning or date/time, etc. '
    )

    commit_hashes = sa.Column(
        sa.ARRAY(sa.String),
        default=[],
        nullable=False,
        doc='List of commit hashes that are included in this version of the code. '
    )

    def update(self):
        repo = git.Repo(search_parent_directories=True)
        git_hash = repo.head.object.hexsha
        if self.commit_hashes is None:
            self.commit_hashes = [git_hash]
        elif git_hash not in self.commit_hashes:
            new_hashes = self.commit_hashes + [git_hash]
            self.commit_hashes = new_hashes


class Provenance(Base):
    __tablename__ = "provenances"

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
        superdict = dict(
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


CodeVersion.metadata.create_all(engine)
Provenance.metadata.create_all(engine)

if __name__ == "__main__":
    pass

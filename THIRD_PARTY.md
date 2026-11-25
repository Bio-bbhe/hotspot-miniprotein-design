# Third-party materials

- The bundled [ProteinMPNN runtime](https://github.com/dauparas/ProteinMPNN)
  is distributed with its [MIT license](dependencies/ProteinMPNN/LICENSE).
  The bundled `v_48_020.pt` is from the upstream ProteinMPNN repository.
- The bundled Transformers 4.57.6 wheel is the archived Biohub fork at
  `ef32577f55da19a4989cd7b22e004dc43a4998cb`, built from that commit before the fork URL
  became unavailable. The Apache 2.0 license is inside the wheel.
  SHA-256: `d277d31e8ccc90ba4fa420e93708da1711e9ba45f328095d55993150b195bfdd`.
- The ESM 3.3.0+release.1 wheel was built from
  [Biohub ESM commit 917af90](https://github.com/Biohub/esm/tree/917af90b624535eed1e072d343c717e3ec11fef4).
  The build changes the version and replaces the unavailable fork URL in
  package metadata with a dependency on the accompanying Transformers
  4.57.6 wheel. The Python source is unchanged. The MIT license is inside the
  wheel. See the [upstream third-party notices](https://github.com/Biohub/esm/blob/917af90b624535eed1e072d343c717e3ec11fef4/THIRD_PARTY_NOTICE.md).
  SHA-256: `9b7477e7def9c388e199d58fe9caee2dd42a135dcfd8ad4e151e5c6ebb34f142`.
- The example target derives from [PDB 4KRL](https://www.rcsb.org/structure/4KRL).
  PDB-derived scaffold entries retain their structure identifiers in the
  filenames. [wwPDB data policy](https://www.wwpdb.org/about/usage-policies).

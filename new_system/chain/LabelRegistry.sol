// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title LabelRegistry
 * @notice Daftar label alamat Ethereum yang terbuka dan dapat diaudit — pelengkap
 *         RiskRegistry.
 *
 * Pembagian peran antara dua kontrak:
 *
 *   RiskRegistry   menyimpan PENDAPAT MODEL  — skor 0–100 hasil machine learning.
 *   LabelRegistry  menyimpan FAKTA & LAPORAN — identitas alamat yang diketahui,
 *                  serta laporan dari pengguna.
 *
 * Latar belakangnya adalah batas nyata yang ditemukan saat pengujian: alamat
 * deposit bursa yang masih baru secara struktur on-chain identik dengan wallet
 * pengumpul phishing (contoh nyata: sebuah hot wallet bursa dengan 17 transaksi
 * masuk dari 16 pengirim dan nol transaksi keluar). Tidak ada model yang bisa
 * memisahkan keduanya dari angka semata — yang membedakan adalah IDENTITAS, dan
 * identitas adalah pengetahuan dari luar rantai.
 *
 * Kontrak ini menyediakan tempat bagi pengetahuan tersebut, dengan dua tingkat:
 *
 *   1. LABEL TERVERIFIKASI  ditetapkan kurator. Mengikat.
 *   2. LAPORAN PENGGUNA     siapa pun boleh melapor. Tidak mengikat; ditampilkan
 *                           sebagai tally suara sampai kurator memutuskan.
 *
 * Pemisahan ini disengaja: pelaporan terbuka mengundang penyalahgunaan, sehingga
 * laporan tidak pernah otomatis menjadi vonis. Yang terbuka adalah datanya —
 * siapa melaporkan apa, kapan — sehingga keputusan kurator bisa diperiksa publik.
 *
 * STATUS: rancangan pendukung sistem baru. Belum di-deploy; RiskRegistry milik
 * sistem tesis tidak diubah sama sekali.
 */
contract LabelRegistry {
    // ── Tipe ─────────────────────────────────────────────────────────────────
    enum Label { Unknown, Safe, Phishing }

    struct Verified {
        Label   label;
        uint64  timestamp;
        string  name;      // mis. "Tokocrypto: Treasury 1"
        string  source;    // mis. "etherscan_label"
    }

    struct Reports {
        uint32 phishingVotes;
        uint32 safeVotes;
    }

    // ── Penyimpanan ──────────────────────────────────────────────────────────
    address public curator;
    mapping(address => Verified) private _verified;
    mapping(address => Reports)  private _reports;
    /// Mencegah satu pelapor menggandakan suaranya untuk alamat yang sama.
    mapping(address => mapping(address => bool)) private _hasReported;

    // ── Event ────────────────────────────────────────────────────────────────
    event LabelVerified(
        address indexed wallet,
        Label           label,
        uint64          timestamp,
        string          name,
        string          source
    );

    event LabelReported(
        address indexed wallet,
        address indexed reporter,
        Label           label,
        uint64          timestamp,
        string          reason
    );

    event CuratorTransferred(address indexed previousCurator, address indexed newCurator);

    // ── Modifier ─────────────────────────────────────────────────────────────
    modifier onlyCurator() {
        require(msg.sender == curator, "LabelRegistry: bukan kurator");
        _;
    }

    constructor() {
        curator = msg.sender;
    }

    // ── Tulis: kurator ───────────────────────────────────────────────────────

    /**
     * @notice Tetapkan label yang mengikat untuk sebuah alamat.
     * @param wallet Alamat yang dilabeli.
     * @param label  Safe atau Phishing (Unknown untuk mencabut label).
     * @param name   Nama yang dikenal publik, mis. "Binance: Hot Wallet".
     * @param source Asal informasinya, mis. "etherscan_label".
     */
    function setVerifiedLabel(
        address wallet,
        Label   label,
        string calldata name,
        string calldata source
    ) external onlyCurator {
        uint64 ts = uint64(block.timestamp);
        _verified[wallet] = Verified(label, ts, name, source);
        emit LabelVerified(wallet, label, ts, name, source);
    }

    /// @notice Menetapkan banyak label sekaligus — untuk memuat daftar awal.
    function setVerifiedLabelBatch(
        address[] calldata wallets,
        Label[]   calldata labels,
        string[]  calldata names,
        string[]  calldata sources
    ) external onlyCurator {
        require(
            wallets.length == labels.length &&
            wallets.length == names.length &&
            wallets.length == sources.length,
            "LabelRegistry: panjang array tidak sama"
        );
        uint64 ts = uint64(block.timestamp);
        for (uint256 i = 0; i < wallets.length; i++) {
            _verified[wallets[i]] = Verified(labels[i], ts, names[i], sources[i]);
            emit LabelVerified(wallets[i], labels[i], ts, names[i], sources[i]);
        }
    }

    // ── Tulis: siapa pun ─────────────────────────────────────────────────────

    /**
     * @notice Laporkan sebuah alamat sebagai phishing atau aman.
     * @dev Laporan TIDAK mengubah label terverifikasi. Ia hanya menambah tally
     *      yang dapat dilihat publik dan dipakai kurator sebagai bahan.
     */
    function reportLabel(address wallet, Label label, string calldata reason) external {
        require(label != Label.Unknown, "LabelRegistry: label harus Safe atau Phishing");
        require(!_hasReported[wallet][msg.sender], "LabelRegistry: sudah pernah melapor");

        _hasReported[wallet][msg.sender] = true;
        if (label == Label.Phishing) {
            _reports[wallet].phishingVotes += 1;
        } else {
            _reports[wallet].safeVotes += 1;
        }
        emit LabelReported(wallet, msg.sender, label, uint64(block.timestamp), reason);
    }

    // ── Baca ─────────────────────────────────────────────────────────────────

    /**
     * @notice Ambil label terverifikasi beserta metadatanya.
     * @return label     Unknown bila belum pernah dilabeli.
     * @return timestamp Waktu pelabelan (0 bila belum ada).
     * @return name      Nama publik.
     * @return source    Asal informasi.
     */
    function getVerifiedLabel(address wallet)
        external
        view
        returns (Label label, uint64 timestamp, string memory name, string memory source)
    {
        Verified storage v = _verified[wallet];
        return (v.label, v.timestamp, v.name, v.source);
    }

    /// @notice Tally laporan pengguna untuk sebuah alamat.
    function getReports(address wallet)
        external
        view
        returns (uint32 phishingVotes, uint32 safeVotes)
    {
        Reports storage r = _reports[wallet];
        return (r.phishingVotes, r.safeVotes);
    }

    /**
     * @notice Keputusan gabungan yang siap dipakai aplikasi.
     * @return label      Label terverifikasi bila ada; selain itu Unknown.
     * @return isVerified True bila berasal dari kurator (mengikat).
     * @return phishingVotes Jumlah laporan phishing dari pengguna.
     * @return safeVotes     Jumlah laporan aman dari pengguna.
     *
     * @dev Sengaja TIDAK menerjemahkan tally menjadi label otomatis. Suara
     *      terbanyak bukan kebenaran, dan pemungutan suara terbuka mudah
     *      dimanipulasi. Aplikasi menampilkan tally sebagai informasi, bukan vonis.
     */
    function resolve(address wallet)
        external
        view
        returns (Label label, bool isVerified, uint32 phishingVotes, uint32 safeVotes)
    {
        Verified storage v = _verified[wallet];
        Reports  storage r = _reports[wallet];
        bool verified = v.label != Label.Unknown;
        return (v.label, verified, r.phishingVotes, r.safeVotes);
    }

    /// @notice Apakah `reporter` sudah pernah melaporkan `wallet`.
    function hasReported(address wallet, address reporter) external view returns (bool) {
        return _hasReported[wallet][reporter];
    }

    // ── Administrasi ─────────────────────────────────────────────────────────
    function transferCuratorship(address newCurator) external onlyCurator {
        require(newCurator != address(0), "LabelRegistry: alamat nol");
        emit CuratorTransferred(curator, newCurator);
        curator = newCurator;
    }
}

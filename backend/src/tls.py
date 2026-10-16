"""
tls.py

Tự sinh chứng chỉ HTTPS tự ký (self-signed) cho mạng LAN.
Trình duyệt điện thoại chỉ cho phép mở camera trên HTTPS (hoặc localhost).
"""

import datetime
import ipaddress
import socket
import time
from pathlib import Path
from typing import List, Tuple


_cache = {"at": 0.0, "ips": []}


def lan_ips() -> List[str]:
    """Các địa chỉ IPv4 LAN của máy (không gồm loopback), cache 30 giây"""
    if time.time() - _cache["at"] < 30:
        return _cache["ips"]
    ips = set()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))  # UDP: không gửi gói tin nào
            ips.add(s.getsockname()[0])
    except OSError:
        pass
    try:
        for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
            ips.add(ip)
    except OSError:
        pass
    _cache["ips"] = sorted(ip for ip in ips if not ip.startswith("127."))
    _cache["at"] = time.time()
    return _cache["ips"]


def ensure_certificate(cert_dir: Path) -> Tuple[str, str]:
    """
    Tạo (hoặc dùng lại) cert/key; tạo lại nếu IP LAN thay đổi

    Returns:
        (cert_path, key_path)
    """
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    cert_dir.mkdir(parents=True, exist_ok=True)
    cert_path, key_path = cert_dir / "cert.pem", cert_dir / "key.pem"
    ips = ["127.0.0.1"] + lan_ips()

    if cert_path.exists() and key_path.exists():
        try:
            cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
            san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
            existing = {str(ip) for ip in san.get_values_for_type(x509.IPAddress)}
            not_after = cert.not_valid_after_utc
            fresh = not_after > datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=7)
            if set(ips) <= existing and fresh:
                return str(cert_path), str(key_path)
        except Exception:
            pass

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Sudoku Solver AI (local)")])
    now = datetime.datetime.now(datetime.timezone.utc)
    san = x509.SubjectAlternativeName(
        [x509.DNSName("localhost"), x509.DNSName(socket.gethostname())]
        + [x509.IPAddress(ipaddress.ip_address(ip)) for ip in ips]
    )
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=825))
        .add_extension(san, critical=False)
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    key_path.write_bytes(key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption(),
    ))
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return str(cert_path), str(key_path)

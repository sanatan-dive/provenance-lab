"""Sign a real photo with a C2PA manifest declaring trainedAlgorithmicMedia (gen-AI),
so the AI-detection branch of check_c2pa is exercised on a genuinely signed file.

Uses the public c2pa-python test ES256 certificate (data/provenance/certs/), so the
signature is cryptographically real but the signer is not on a trust list.

Run:  .venv/bin/python -m provenance.make_c2pa_sample
"""
import c2pa

D = "data/provenance/"

MANIFEST = {
    "claim_generator_info": [{"name": "cs756_provenance_demo", "version": "0.1"}],
    "title": "AI-generated demo",
    "assertions": [{
        "label": "c2pa.actions",
        "data": {"actions": [{
            "action": "c2pa.created",
            "digitalSourceType": "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia",
            "softwareAgent": "cs756 demo generator",
        }]},
    }],
}


def sign(src, dst):
    info = c2pa.C2paSignerInfo(
        b"es256", open(D + "certs/es256_certs.pem", "rb").read(),
        open(D + "certs/es256_private.key", "rb").read(), None)
    with c2pa.Signer.from_info(info) as signer, c2pa.Builder(MANIFEST) as builder:
        builder.sign_file(src, dst, signer)
    print("signed", dst)


if __name__ == "__main__":
    sign(D + "A.jpg", D + "A_ai_signed.jpg")

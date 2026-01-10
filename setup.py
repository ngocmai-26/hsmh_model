import setuptools

with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip()]

setuptools.setup(
    name="hsmh_model",
    version="0.0.1",
    # Chỉ lấy các package thuộc namespace hsmh_model (vd: hsmh_model, hsmh_model.model, ...)
    packages=[
        pkg for pkg in setuptools.find_packages()
        if pkg == "hsmh_model" or pkg.startswith("hsmh_model.")
    ],
    include_package_data=True,
    package_data={
        'hsmh_model': [
            'trained_models/class_model/*.pkl', 
            'trained_models/individual_model/*.pkl',
        ],
    },
    install_requires=requirements,
    python_requires='>=3.8',
)

import { ethers } from "hardhat";
import * as fs from "fs";
import * as path from "path";

async function main() {
  console.log("Deploying RiskRegistry…");

  const [deployer] = await ethers.getSigners();
  console.log(`Deployer: ${deployer.address}`);
  console.log(
    `Deployer balance: ${ethers.formatEther(
      await ethers.provider.getBalance(deployer.address)
    )} ETH`
  );

  const RiskRegistry = await ethers.getContractFactory("RiskRegistry");
  const registry = await RiskRegistry.deploy();
  await registry.waitForDeployment();

  const address = await registry.getAddress();
  console.log(`RiskRegistry deployed to: ${address}`);

  // ── Persist ABI + address for Python web3.py ──────────────────────────────
  const artifactPath = path.join(
    __dirname,
    "..",
    "artifacts",
    "contracts",
    "RiskRegistry.sol",
    "RiskRegistry.json"
  );

  const deploymentInfo = {
    address,
    deployer: deployer.address,
    network: "localhost",
    chainId: 31337,
  };

  // Write deployment info next to the ABI artifact directory
  const outDir = path.join(__dirname, "..", "artifacts");
  fs.mkdirSync(outDir, { recursive: true });
  const deployPath = path.join(outDir, "deployment.json");
  fs.writeFileSync(deployPath, JSON.stringify(deploymentInfo, null, 2));
  console.log(`Deployment info written to: ${deployPath}`);

  // Also copy ABI to a flat file for easy Python import
  if (fs.existsSync(artifactPath)) {
    const artifact = JSON.parse(fs.readFileSync(artifactPath, "utf8"));
    const abiPath = path.join(outDir, "RiskRegistry.abi.json");
    fs.writeFileSync(abiPath, JSON.stringify(artifact.abi, null, 2));
    console.log(`ABI written to: ${abiPath}`);
  }
}

main()
  .then(() => process.exit(0))
  .catch((err) => {
    console.error(err);
    process.exit(1);
  });

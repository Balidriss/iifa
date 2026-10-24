// iifa memory for opencode. Copy this file and the iifa/ directory next to it
// into .opencode/plugins/ (or ~/.config/opencode/plugins/). opencode loads every
// file of that directory as a plugin, not its subdirectories: the logic lives in
// iifa/core.ts. See adapters/opencode/README.md.
import { tool } from "@opencode-ai/plugin"
import { createServer } from "./iifa/core.ts"

export default { id: "iifa", server: createServer({ schema: tool.schema }) }
